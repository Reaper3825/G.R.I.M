#include "Batching_GPU.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <random>

namespace GRIM::Batching {

// =============================================================================
// Internal Helpers
// =============================================================================
namespace {

inline uint32_t bucketize(uint32_t len, uint32_t step) {
    step = std::max(step, 1u);
    return (len + step - 1) / step;
}

// Compute variance of lengths in a batch
inline float computeLengthVariance(const std::vector<uint32_t>& lengths) {
    if (lengths.size() < 2) return 0.0f;
    float mean = 0.0f;
    for (auto l : lengths) mean += static_cast<float>(l);
    mean /= lengths.size();
    float var = 0.0f;
    for (auto l : lengths) {
        float diff = static_cast<float>(l) - mean;
        var += diff * diff;
    }
    return var / lengths.size();
}

// Check if two lengths are "similar" within threshold
inline bool areSimilarLengths(uint32_t len1, uint32_t len2, float threshold) {
    if (len1 == 0 || len2 == 0) return false;
    uint32_t max_len = std::max(len1, len2);
    uint32_t min_len = std::min(len1, len2);
    float ratio = static_cast<float>(max_len - min_len) / static_cast<float>(max_len);
    return ratio <= threshold;
}

// Compute percentile from sorted lengths
inline uint32_t percentile(const std::vector<uint32_t>& sorted_lengths, float p) {
    if (sorted_lengths.empty()) return 0;
    size_t idx = static_cast<size_t>(p * (sorted_lengths.size() - 1));
    return sorted_lengths[std::min(idx, sorted_lengths.size() - 1)];
}

// Finalize batch assignment statistics
inline void finalizeBatchStats(BatchAssignment& batch, const std::vector<uint32_t>& lengths) {
    if (batch.seq_ids.empty()) return;
    
    batch.total_tokens = batch.max_seq_len * static_cast<uint32_t>(batch.seq_ids.size());
    if (batch.total_tokens >= batch.actual_tokens) {
        batch.padding_tokens = batch.total_tokens - batch.actual_tokens;
    }
    batch.packing_efficiency = batch.total_tokens > 0 
        ? static_cast<float>(batch.actual_tokens) / batch.total_tokens 
        : 0.0f;
    batch.length_variance = computeLengthVariance(lengths);
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
    ss << "  Batch size range: " << min_batch_size_observed << "-" << max_batch_size_observed;
    ss << " (avg: " << avg_batch_size << ")\n";
    ss << "  Tokens: " << actual_tokens << " actual / " << total_tokens << " compute";
    ss << " (padding: " << padding_tokens << ", " << (100.0f * padding_tokens / (total_tokens > 0 ? total_tokens : uint64_t(1))) << "%)\n";
    ss << "  Packing efficiency: " << (100.0f * avg_packing_efficiency) << "%\n";
    ss << "  Overflow batches: " << overflow_batches << "\n";
    ss << "  Max seq len: " << max_seq_len_observed << "\n";
    ss << "  Seq len percentiles: p50=" << p50_seq_len << " p90=" << p90_seq_len << " p99=" << p99_seq_len << "\n";
    return ss.str();
}

// =============================================================================
// Build Batches - Main Implementation
// =============================================================================
BatchSchedule buildBatches(const Catalog& catalog, const BatchOptions& opts) {
    BatchSchedule schedule{};
    
    if (catalog.entries().empty()) {
        return schedule;
    }
    if (opts.max_tokens_per_batch == 0) {
        throw std::runtime_error("buildBatches: max_tokens_per_batch=0 — caller MUST set token budget");
    }
    if (opts.max_batch_size == 0) {
        throw std::runtime_error("buildBatches: max_batch_size=0 — caller MUST set batch size limit");
    }

    const auto& entries = catalog.entries();
    const uint32_t bucket_step = std::max(1u, opts.bucket_step);
    
    // Build indices of valid sequences + compute length percentiles in ONE pass
    std::vector<uint32_t> view_indices;
    std::vector<uint32_t> all_lengths;
    
    view_indices.reserve(entries.size());
    all_lengths.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].seq_length > 0) {
            view_indices.push_back(static_cast<uint32_t>(i));
            all_lengths.push_back(entries[i].seq_length);
        }
    }

    if (view_indices.empty()) return schedule;
    
    // Compute length percentiles for schedule stats (ALL lengths, not just finalized batches)
    std::vector<uint32_t> sorted_lengths = all_lengths;
    std::sort(sorted_lengths.begin(), sorted_lengths.end());
    schedule.p50_seq_len = percentile(sorted_lengths, 0.50f);
    schedule.p90_seq_len = percentile(sorted_lengths, 0.90f);
    schedule.p99_seq_len = percentile(sorted_lengths, 0.99f);
    
    // Lambda to compute sort key for a sequence by index
    auto computeSortKey = [&](uint32_t entry_idx) -> float {
        const uint32_t len = entries[entry_idx].seq_length;
        float len_key = opts.prefer_short_first ? static_cast<float>(len) : -static_cast<float>(len);
        
        // Apply curriculum: filter long sequences early in training
        if (opts.curriculum_progress < 1.0f) {
            uint32_t max_allowed = static_cast<uint32_t>(
                schedule.p50_seq_len + opts.curriculum_progress * (schedule.p99_seq_len - schedule.p50_seq_len)
            );
            if (len > max_allowed) {
                len_key += 1e6f;  // push to end
            }
        }
        return len_key;
    };
    
    // Sort indices based on strategy
    if (opts.strategy == PackingStrategy::SIMILARITY_GROUPED) {
        // Sort by bucket (groups similar lengths together)
        std::stable_sort(view_indices.begin(), view_indices.end(),
            [&](uint32_t idx_a, uint32_t idx_b) {
                uint32_t bucket_a = bucketize(entries[idx_a].seq_length, bucket_step);
                uint32_t bucket_b = bucketize(entries[idx_b].seq_length, bucket_step);
                if (bucket_a != bucket_b) {
                    return opts.prefer_short_first ? (bucket_a < bucket_b) : (bucket_a > bucket_b);
                }
                return computeSortKey(idx_a) < computeSortKey(idx_b);
            });
    } else if (opts.strategy == PackingStrategy::BEST_FIT_DECREASING) {
        // Sort longest first (FFD algorithm)
        std::stable_sort(view_indices.begin(), view_indices.end(),
            [&](uint32_t idx_a, uint32_t idx_b) {
                return entries[idx_a].seq_length > entries[idx_b].seq_length;
            });
    } else {
        // GREEDY / Gradient-balanced: SHUFFLE to mix short and long sequences!
        // Issue #90: Length-sorted batching caused mode collapse at max_seq_len boundary.
        // By shuffling, we expose ALL position ranges from batch 1, preventing the
        // boundary effect where positions 671-1023 were only first seen in batch 5.
        if (opts.rng_seed != 0) {
            // Deterministic shuffle using provided seed
            std::mt19937_64 shuffle_rng(opts.rng_seed);
            std::shuffle(view_indices.begin(), view_indices.end(), shuffle_rng);
        } else {
            // Non-deterministic shuffle (fallback)
            std::random_device rd;
            std::mt19937_64 shuffle_rng(rd());
            std::shuffle(view_indices.begin(), view_indices.end(), shuffle_rng);
        }

        // Apply curriculum / prefer_short_first ordering on top of the shuffle.
        // stable_sort preserves the shuffled order among entries with equal sort
        // keys, so we still mix sequences within each curriculum tier rather than
        // reverting to a strict length-sorted order. Without this, computeSortKey()
        // (and therefore curriculum_progress / prefer_short_first) is never used
        // on the GREEDY path, which is the path training selects.
        const bool curriculum_active =
            opts.prefer_short_first ||
            (opts.curriculum_progress < 1.0f &&
             schedule.p99_seq_len > schedule.p50_seq_len);
        if (curriculum_active) {
            std::stable_sort(view_indices.begin(), view_indices.end(),
                [&](uint32_t idx_a, uint32_t idx_b) {
                    return computeSortKey(idx_a) < computeSortKey(idx_b);
                });
        }
    }
    
    // Use pre-computed token budget from analyzeAndRecommend() (Phase2 sets opts.max_tokens_per_batch)
    const uint32_t token_budget = opts.max_tokens_per_batch;
    
    // =======================================================================
    // Packing algorithm
    // =======================================================================
    
    if (opts.strategy == PackingStrategy::BEST_FIT_DECREASING) {
        // FFD: For each sequence, find the batch where it fits best (least waste)
        // Open batches that we can still add to
        struct OpenBatch {
            BatchAssignment assignment;
            std::vector<uint32_t> lengths;
            uint32_t remaining_slots;
        };
        std::vector<OpenBatch> open_batches;
        
        for (uint32_t entry_idx : view_indices) {
            const uint32_t seq_len = entries[entry_idx].seq_length;
            const uint32_t seq_id = entries[entry_idx].seq_id;
            
            // Check for overflow
            if (seq_len > token_budget) {
                BatchAssignment overflow{};
                overflow.seq_ids.push_back(seq_id);
                overflow.max_seq_len = seq_len;
                overflow.min_seq_len = seq_len;
                overflow.actual_tokens = seq_len;
                overflow.overflow = true;
                finalizeBatchStats(overflow, {seq_len});
                schedule.batches.push_back(std::move(overflow));
                schedule.overflow_batches++;
                continue;
            }
            
            // Find best-fit batch (minimize waste increase)
            int best_batch = -1;
            uint32_t best_waste_increase = UINT32_MAX;
            
            for (size_t i = 0; i < open_batches.size(); ++i) {
                auto& ob = open_batches[i];
                if (ob.remaining_slots == 0) continue;
                
                uint32_t new_max = std::max(ob.assignment.max_seq_len, seq_len);
                uint32_t new_size = static_cast<uint32_t>(ob.assignment.seq_ids.size()) + 1;
                // Use uint64_t to avoid uint32_t wrap on max_seq_len * batch_size.
                uint64_t new_total = static_cast<uint64_t>(new_max) * new_size;

                if (new_total > static_cast<uint64_t>(token_budget) ||
                    new_size > opts.max_batch_size) continue;
                
                // Check similarity constraint if enabled
                if (opts.similarity_threshold < 1.0f) {
                    if (!areSimilarLengths(ob.assignment.min_seq_len, seq_len, opts.similarity_threshold) ||
                        !areSimilarLengths(ob.assignment.max_seq_len, seq_len, opts.similarity_threshold)) {
                        continue;
                    }
                }
                
                uint64_t old_total = ob.assignment.total_tokens;
                // new_total >= old_total + seq_len by construction (new_max >= old_max,
                // new_size = old_size + 1), so this subtraction is safe in uint64_t.
                uint64_t waste_increase_u64 = new_total - old_total - seq_len;
                uint32_t waste_increase = waste_increase_u64 > UINT32_MAX
                    ? UINT32_MAX
                    : static_cast<uint32_t>(waste_increase_u64);

                if (waste_increase < best_waste_increase) {
                    best_waste_increase = waste_increase;
                    best_batch = static_cast<int>(i);
                }
            }
            
            if (best_batch >= 0) {
                // Add to existing batch
                auto& ob = open_batches[best_batch];
                ob.assignment.seq_ids.push_back(seq_id);
                ob.assignment.max_seq_len = std::max(ob.assignment.max_seq_len, seq_len);
                ob.assignment.min_seq_len = std::min(ob.assignment.min_seq_len, seq_len);
                ob.assignment.actual_tokens += seq_len;
                {
                    uint64_t total64 = static_cast<uint64_t>(ob.assignment.max_seq_len) *
                                       static_cast<uint64_t>(ob.assignment.seq_ids.size());
                    ob.assignment.total_tokens = total64 > UINT32_MAX
                        ? UINT32_MAX
                        : static_cast<uint32_t>(total64);
                }
                ob.lengths.push_back(seq_len);
                ob.remaining_slots--;

                // Close the batch only when no more sequences can fit. Previously
                // we also closed at packing efficiency >= 0.9, which caused two
                // equal-length sequences to seal a batch at 100% efficiency even
                // when both max_batch_size and the token budget could fit many
                // more — turning four length-100 seqs (budget 400, max 4) into
                // two batches of 2 instead of one batch of 4.
                const uint64_t budget_close_threshold =
                    static_cast<uint64_t>(token_budget) * 85ULL / 100ULL;
                if (ob.remaining_slots == 0 ||
                    static_cast<uint64_t>(ob.assignment.total_tokens) >= budget_close_threshold) {
                    finalizeBatchStats(ob.assignment, ob.lengths);
                    schedule.batches.push_back(std::move(ob.assignment));
                    open_batches.erase(open_batches.begin() + best_batch);
                }
            } else {
                // Start new batch
                OpenBatch new_batch;
                new_batch.assignment.seq_ids.push_back(seq_id);
                new_batch.assignment.max_seq_len = seq_len;
                new_batch.assignment.min_seq_len = seq_len;
                new_batch.assignment.actual_tokens = seq_len;
                new_batch.assignment.total_tokens = seq_len;
                new_batch.lengths.push_back(seq_len);
                new_batch.remaining_slots = opts.max_batch_size - 1;
                open_batches.push_back(std::move(new_batch));
            }
        }
        
        // Finalize remaining open batches
        for (auto& ob : open_batches) {
            if (!ob.assignment.seq_ids.empty()) {
                finalizeBatchStats(ob.assignment, ob.lengths);
                schedule.batches.push_back(std::move(ob.assignment));
            }
        }
        
    } else {
        // GREEDY / SIMILARITY_GROUPED
        // Standard first-fit with similarity constraints
        
        BatchAssignment current{};
        std::vector<uint32_t> current_lengths;
        
        auto finalizeCurrent = [&]() {
            if (current.seq_ids.empty()) return;
            finalizeBatchStats(current, current_lengths);
            schedule.batches.push_back(std::move(current));
            current = BatchAssignment{};
            current_lengths.clear();
        };
        
        for (uint32_t entry_idx : view_indices) {
            const uint32_t seq_len = entries[entry_idx].seq_length;
            const uint32_t seq_id = entries[entry_idx].seq_id;
            
            // Check for overflow
            if (seq_len > token_budget) {
                finalizeCurrent();
                BatchAssignment overflow{};
                overflow.seq_ids.push_back(seq_id);
                overflow.max_seq_len = seq_len;
                overflow.min_seq_len = seq_len;
                overflow.actual_tokens = seq_len;
                overflow.overflow = true;
                finalizeBatchStats(overflow, {seq_len});
                schedule.batches.push_back(std::move(overflow));
                schedule.overflow_batches++;
                continue;
            }
            
            uint32_t prospective_size = static_cast<uint32_t>(current.seq_ids.size()) + 1;
            uint32_t prospective_max = std::max(current.max_seq_len, seq_len);
            // Use uint64_t to avoid uint32_t wrap on max_seq_len * batch_size.
            uint64_t prospective_tokens =
                static_cast<uint64_t>(prospective_size) * static_cast<uint64_t>(prospective_max);

            bool exceeds_budget = prospective_tokens > static_cast<uint64_t>(token_budget);
            bool exceeds_size = prospective_size > opts.max_batch_size;
            bool breaks_similarity = false;
            
            if (opts.strategy == PackingStrategy::SIMILARITY_GROUPED && !current.seq_ids.empty()) {
                breaks_similarity = !areSimilarLengths(current.min_seq_len, seq_len, opts.similarity_threshold) ||
                                   !areSimilarLengths(current.max_seq_len, seq_len, opts.similarity_threshold);
            }
            
            if (!current.seq_ids.empty() && (exceeds_budget || exceeds_size || breaks_similarity)) {
                finalizeCurrent();
            }
            
            // Add to current batch
            current.seq_ids.push_back(seq_id);
            current.max_seq_len = std::max(current.max_seq_len, seq_len);
            current.min_seq_len = current.min_seq_len == 0 ? seq_len : std::min(current.min_seq_len, seq_len);
            current.actual_tokens += seq_len;
            {
                uint64_t total64 = static_cast<uint64_t>(current.max_seq_len) *
                                   static_cast<uint64_t>(current.seq_ids.size());
                current.total_tokens = total64 > UINT32_MAX
                    ? UINT32_MAX
                    : static_cast<uint32_t>(total64);
            }
            current_lengths.push_back(seq_len);
        }
        
        finalizeCurrent();
    }
    
    // =======================================================================
    // Compute schedule statistics
    // =======================================================================
    
    uint64_t total_actual = 0;
    uint64_t total_compute = 0;
    float efficiency_sum = 0.0f;
    uint64_t batch_size_sum = 0;
    
    for (const auto& batch : schedule.batches) {
        total_actual += batch.actual_tokens;
        total_compute += batch.total_tokens;
        efficiency_sum += batch.packing_efficiency;
        batch_size_sum += batch.seq_ids.size();
        
        uint32_t size = static_cast<uint32_t>(batch.seq_ids.size());
        schedule.min_batch_size_observed = schedule.min_batch_size_observed == 0 
            ? size : std::min(schedule.min_batch_size_observed, size);
        schedule.max_batch_size_observed = std::max(schedule.max_batch_size_observed, size);
        schedule.max_seq_len_observed = std::max(schedule.max_seq_len_observed, batch.max_seq_len);
    }
    
    schedule.actual_tokens = total_actual;
    schedule.total_tokens = total_compute;
    schedule.padding_tokens = total_compute - total_actual;
    
    if (!schedule.batches.empty()) {
        schedule.avg_batch_size = static_cast<float>(batch_size_sum) / schedule.batches.size();
        schedule.avg_packing_efficiency = efficiency_sum / schedule.batches.size();
    }
    
    // =======================================================================
    // Post-packing batch ordering (curriculum learning)
    // =======================================================================
    
    if (opts.batch_ordering != BatchOrdering::PRESERVE && schedule.batches.size() > 1) {
        // Separate overflow batches if requested
        std::vector<BatchAssignment> overflow_batches;
        std::vector<BatchAssignment> normal_batches;
        
        if (opts.interleave_overflow) {
            for (auto& batch : schedule.batches) {
                if (batch.overflow) {
                    overflow_batches.push_back(std::move(batch));
                } else {
                    normal_batches.push_back(std::move(batch));
                }
            }
        } else {
            normal_batches = std::move(schedule.batches);
        }
        
        // Sort normal batches based on ordering strategy
        auto getAvgLen = [](const BatchAssignment& b) -> float {
            return b.seq_ids.empty() ? 0.0f : static_cast<float>(b.actual_tokens) / b.seq_ids.size();
        };
        
        switch (opts.batch_ordering) {
            case BatchOrdering::LENGTH_ASCENDING:
                std::stable_sort(normal_batches.begin(), normal_batches.end(),
                    [&](const BatchAssignment& a, const BatchAssignment& b) {
                        return getAvgLen(a) < getAvgLen(b);
                    });
                break;
                
            case BatchOrdering::LENGTH_DESCENDING:
                std::stable_sort(normal_batches.begin(), normal_batches.end(),
                    [&](const BatchAssignment& a, const BatchAssignment& b) {
                        return getAvgLen(a) > getAvgLen(b);
                    });
                break;
                
            case BatchOrdering::INTERLEAVED: {
                // Sort by length, then interleave from both ends
                std::stable_sort(normal_batches.begin(), normal_batches.end(),
                    [&](const BatchAssignment& a, const BatchAssignment& b) {
                        return getAvgLen(a) < getAvgLen(b);
                    });
                std::vector<BatchAssignment> interleaved;
                interleaved.reserve(normal_batches.size());
                size_t left = 0, right = normal_batches.size() - 1;
                bool pick_left = true;
                while (left <= right && right < normal_batches.size()) {
                    if (pick_left) {
                        interleaved.push_back(std::move(normal_batches[left++]));
                    } else {
                        interleaved.push_back(std::move(normal_batches[right--]));
                    }
                    pick_left = !pick_left;
                }
                normal_batches = std::move(interleaved);
                break;
            }
                
            case BatchOrdering::RANDOM: {
                // Fisher-Yates shuffle — use opts.rng_seed for reproducibility.
                // Guard against size() == 0: with size_t, `size() - 1` underflows
                // to SIZE_MAX. This path is reachable when interleave_overflow=true
                // moves every batch into overflow_batches, leaving normal_batches
                // empty.
                if (normal_batches.size() > 1) {
                    const uint64_t order_seed = (opts.rng_seed != 0)
                        ? opts.rng_seed + 0x9E3779B97F4A7C15ULL  // derive distinct seed from packing seed
                        : std::random_device{}();
                    std::mt19937_64 rng(order_seed);
                    for (size_t i = normal_batches.size() - 1; i > 0; --i) {
                        std::uniform_int_distribution<size_t> dist(0, i);
                        std::swap(normal_batches[i], normal_batches[dist(rng)]);
                    }
                }
                break;
            }
                
            case BatchOrdering::PRESERVE:
            default:
                break;
        }
        
        // Reintegrate overflow batches if they were separated
        if (opts.interleave_overflow && !overflow_batches.empty()) {
            // Sort overflow by length too
            std::stable_sort(overflow_batches.begin(), overflow_batches.end(),
                [&](const BatchAssignment& a, const BatchAssignment& b) {
                    return getAvgLen(a) < getAvgLen(b);
                });
            
            // Distribute overflow batches evenly throughout normal batches
            schedule.batches.clear();
            schedule.batches.reserve(normal_batches.size() + overflow_batches.size());
            
            if (normal_batches.empty()) {
                schedule.batches = std::move(overflow_batches);
            } else {
                float interval = static_cast<float>(normal_batches.size()) / (overflow_batches.size() + 1);
                size_t overflow_idx = 0;
                float next_overflow_pos = interval;
                
                for (size_t i = 0; i < normal_batches.size(); ++i) {
                    // Insert overflow batch at calculated positions
                    while (overflow_idx < overflow_batches.size() && 
                           static_cast<float>(i) >= next_overflow_pos) {
                        schedule.batches.push_back(std::move(overflow_batches[overflow_idx++]));
                        next_overflow_pos += interval;
                    }
                    schedule.batches.push_back(std::move(normal_batches[i]));
                }
                // Add remaining overflow batches at the end
                while (overflow_idx < overflow_batches.size()) {
                    schedule.batches.push_back(std::move(overflow_batches[overflow_idx++]));
                }
            }
        } else {
            schedule.batches = std::move(normal_batches);
        }
    }
    
    
    return schedule;
}

} // namespace GRIM::Batching
