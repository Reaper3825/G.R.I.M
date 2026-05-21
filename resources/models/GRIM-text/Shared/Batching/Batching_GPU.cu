#include "Batching_GPU.hpp"
#include "PackerPolicy.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace GRIM::Batching {

// =============================================================================
// Internal Helpers
// =============================================================================
namespace {

// Compute percentile from sorted lengths
inline uint32_t percentile(const std::vector<uint32_t>& sorted_lengths, float p) {
    if (sorted_lengths.empty()) return 0;
    size_t idx = static_cast<size_t>(p * (sorted_lengths.size() - 1));
    return sorted_lengths[std::min(idx, sorted_lengths.size() - 1)];
}

inline uint64_t sumActualTokens(
    const BatchAssignment& batch,
    const std::vector<uint32_t>& sequence_lengths)
{
    uint64_t total = 0;
    for (uint32_t seq_id : batch.seq_ids) {
        if (seq_id >= sequence_lengths.size()) {
            throw std::runtime_error(
                "buildBatches: batch seq_id=" + std::to_string(seq_id) +
                " exceeds sequence_lengths.size()=" + std::to_string(sequence_lengths.size()));
        }
        total += sequence_lengths[seq_id];
    }
    return total;
}

inline uint32_t maxObservedLength(
    const BatchAssignment& batch,
    const std::vector<uint32_t>& sequence_lengths)
{
    uint32_t max_len = 0;
    for (uint32_t seq_id : batch.seq_ids) {
        if (seq_id >= sequence_lengths.size()) {
            throw std::runtime_error(
                "buildBatches: batch seq_id=" + std::to_string(seq_id) +
                " exceeds sequence_lengths.size()=" + std::to_string(sequence_lengths.size()));
        }
        max_len = std::max(max_len, sequence_lengths[seq_id]);
    }
    return max_len;
}

} // anonymous namespace

// =============================================================================
// Schedule Summary
// =============================================================================
std::string BatchSchedule::summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "BatchSchedule Summary:\n";
    ss << "  Total batches: " << batches.size() << "\n";
    ss << "  Batch size: " << batch_size << "\n";
    ss << "  Discarded tail sequences: " << discarded_tail_sequences << "\n";
    ss << "  Tokens: " << actual_tokens << " actual / " << total_tokens << " compute";
    ss << " (padding: " << padding_tokens << ", " << (100.0f * padding_tokens / (total_tokens > 0 ? total_tokens : uint64_t(1))) << "%)\n";
    ss << "  Packing efficiency: " << (100.0f * packing_efficiency) << "%\n";
    ss << "  Sequence cap: " << sequence_cap << "\n";
    ss << "  Max observed seq len: " << max_seq_len_observed << "\n";
    ss << "  Seq len percentiles: p50=" << p50_seq_len << " p90=" << p90_seq_len << " p99=" << p99_seq_len << "\n";
    return ss.str();
}

// =============================================================================
// Build Batches - Main Implementation
// =============================================================================
BatchSchedule buildBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t fixed_sequence_cap,
    uint32_t fixed_batch_size,
    const PackerPolicy& policy)
{
    BatchSchedule schedule{};

    if (fixed_sequence_cap == 0) {
        throw std::runtime_error("buildBatches: fixed_sequence_cap=0 — caller MUST pass configured max_seq_len");
    }
    if (fixed_batch_size == 0) {
        throw std::runtime_error("buildBatches: fixed_batch_size=0 — caller MUST pass configured batch_size");
    }

    schedule.sequence_cap = fixed_sequence_cap;
    schedule.batch_size = fixed_batch_size;

    if (sequence_lengths.empty()) {
        return schedule;
    }

    // Build indices of valid sequences + compute length percentiles in ONE pass
    std::vector<uint32_t> view_indices;
    std::vector<uint32_t> all_lengths;
    
    view_indices.reserve(sequence_lengths.size());
    all_lengths.reserve(sequence_lengths.size());

    for (size_t i = 0; i < sequence_lengths.size(); ++i) {
        if (sequence_lengths[i] > 0) {
            view_indices.push_back(static_cast<uint32_t>(i));
            all_lengths.push_back(sequence_lengths[i]);
        }
    }

    if (view_indices.empty()) return schedule;
    
    // Compute length percentiles for schedule stats (ALL lengths, not just finalized batches)
    std::vector<uint32_t> sorted_lengths = all_lengths;
    std::sort(sorted_lengths.begin(), sorted_lengths.end());
    schedule.p50_seq_len = percentile(sorted_lengths, 0.50f);
    schedule.p90_seq_len = percentile(sorted_lengths, 0.90f);
    schedule.p99_seq_len = percentile(sorted_lengths, 0.99f);
    
    // Deterministic shuffle to mix sequence lengths when training requests it.
    // Zero seed means preserve input order, which validation relies on.
    if (policy.rng_seed != 0) {
        std::mt19937_64 shuffle_rng(policy.rng_seed);
        std::shuffle(view_indices.begin(), view_indices.end(), shuffle_rng);
    }
    
    // =======================================================================
    // Fixed-row packing algorithm
    // =======================================================================
    
    BatchAssignment current{};
    current.seq_ids.reserve(fixed_batch_size);

    for (uint32_t entry_idx : view_indices) {
        const uint32_t seq_len = sequence_lengths[entry_idx];
        const uint32_t seq_id = entry_idx;
        
        // Capacity contract (Rule 20): no overflow batches. If a single
        // sequence violates the fixed sequence cap, fail loud with identifiers.
        if (seq_len > fixed_sequence_cap) {
            throw std::runtime_error(
                "buildBatches: sequence exceeds fixed sequence cap (seq_id=" + std::to_string(seq_id) +
                " seq_len=" + std::to_string(seq_len) +
                " sequence_cap=" + std::to_string(fixed_sequence_cap) + ")");
        }

        current.seq_ids.push_back(seq_id);
        if (current.seq_ids.size() == fixed_batch_size) {
            schedule.batches.push_back(std::move(current));
            current = BatchAssignment{};
            current.seq_ids.reserve(fixed_batch_size);
        }
    }

    if (!current.seq_ids.empty()) {
        schedule.discarded_tail_sequences = static_cast<uint32_t>(current.seq_ids.size());
    }
    
    // =======================================================================
    // Compute schedule statistics
    // =======================================================================
    
    uint64_t total_actual = 0;
    uint64_t total_compute = 0;
    
    for (const auto& batch : schedule.batches) {
        const uint32_t size = static_cast<uint32_t>(batch.seq_ids.size());
        if (size != fixed_batch_size) {
            throw std::runtime_error(
                "buildBatches: non-fixed batch size emitted (batch_size=" +
                std::to_string(size) + " expected=" + std::to_string(fixed_batch_size) +
                ") — fixed batch size is required; upstream data count/order must produce full batches");
        }
        const uint64_t batch_actual = sumActualTokens(batch, sequence_lengths);
        const uint64_t batch_compute = static_cast<uint64_t>(fixed_sequence_cap) * fixed_batch_size;
        if (batch_compute < batch_actual) {
            throw std::runtime_error(
                "buildBatches: fixed compute tokens smaller than actual tokens (batch_compute=" +
                std::to_string(batch_compute) + " batch_actual=" +
                std::to_string(batch_actual) + " sequence_cap=" +
                std::to_string(fixed_sequence_cap) + " batch_size=" +
                std::to_string(batch.seq_ids.size()) + ")");
        }
        total_actual += batch_actual;
        total_compute += batch_compute;
        schedule.max_seq_len_observed = std::max(
            schedule.max_seq_len_observed,
            maxObservedLength(batch, sequence_lengths));
    }
    
    schedule.actual_tokens = total_actual;
    schedule.total_tokens = total_compute;
    schedule.padding_tokens = total_compute - total_actual;
    
    if (total_compute > 0) {
        schedule.packing_efficiency = static_cast<float>(total_actual) / static_cast<float>(total_compute);
    }
    
    // =======================================================================
    // Post-packing batch ordering
    // =======================================================================
    
    if (policy.batch_ordering != BatchOrdering::PRESERVE && schedule.batches.size() > 1) {
        std::vector<BatchAssignment> normal_batches = std::move(schedule.batches);
        
        switch (policy.batch_ordering) {
            case BatchOrdering::RANDOM: {
                // Fisher-Yates shuffle — use policy.rng_seed for reproducibility.
                if (normal_batches.size() > 1) {
                    const uint64_t order_seed = (policy.rng_seed != 0)
                        ? policy.rng_seed + 0x9E3779B97F4A7C15ULL  // derive distinct seed from packing seed
                        : 0xD1B54A32D192ED03ULL;
                    std::mt19937_64 rng(order_seed);
                    for (size_t i = normal_batches.size() - 1; i > 0; --i) {
                        std::uniform_int_distribution<size_t> dist(0, i);
                        std::swap(normal_batches[i], normal_batches[dist(rng)]);
                    }
                }
                break;
            }
                
            case BatchOrdering::PRESERVE:
                break;
        }
        schedule.batches = std::move(normal_batches);
    }
    
    
    return schedule;
}

} // namespace GRIM::Batching
