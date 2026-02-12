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

struct SequenceView {
    uint32_t seq_id;
    uint32_t length;
    uint32_t bucket_key;
    float sort_key;  // combined sorting metric
};

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
    ss << " (padding: " << padding_tokens << ", " << (100.0f * padding_tokens / std::max(1ULL, total_tokens)) << "%)\n";
    ss << "  Packing efficiency: " << (100.0f * avg_packing_efficiency) << "%\n";
    ss << "  Overflow batches: " << overflow_batches << "\n";
    ss << "  Max seq len: " << max_seq_len_observed << "\n";
    ss << "  Seq len percentiles: p50=" << p50_seq_len << " p90=" << p90_seq_len << " p99=" << p99_seq_len << "\n";
    if (num_accumulation_groups > 0) {
        ss << "  Gradient accumulation: " << num_accumulation_groups << " groups";
        ss << " (effective batch: " << effective_batch_size << ")\n";
    }
    return ss.str();
}

// =============================================================================
// Analyze and Recommend
// =============================================================================
BatchOptions analyzeAndRecommend(const Catalog& catalog, uint32_t target_effective_batch) {
    BatchOptions opts;
    
    const auto& entries = catalog.entries();
    if (entries.empty()) return opts;
    
    // Collect and sort lengths
    std::vector<uint32_t> lengths;
    lengths.reserve(entries.size());
    for (const auto& e : entries) {
        if (e.seq_length > 0) lengths.push_back(e.seq_length);
    }
    if (lengths.empty()) return opts;
    
    std::sort(lengths.begin(), lengths.end());
    
    // Compute statistics
    uint32_t p50 = percentile(lengths, 0.50f);
    uint32_t p75 = percentile(lengths, 0.75f);
    uint32_t p90 = percentile(lengths, 0.90f);
    uint32_t p99 = percentile(lengths, 0.99f);
    uint32_t max_len = lengths.back();
    
    uint64_t sum = 0;
    for (auto l : lengths) sum += l;
    float mean = static_cast<float>(sum) / lengths.size();
    
    // Recommend token budget: aim for 2-4 average sequences per batch
    // Use p75 as reference to handle typical sequences well
    uint32_t recommended_budget = std::max(p75 * 3, p50 * 4);
    recommended_budget = std::max(recommended_budget, 2048u);  // minimum floor
    recommended_budget = std::min(recommended_budget, 16384u); // maximum cap
    
    // Round to nice number
    recommended_budget = ((recommended_budget + 511) / 512) * 512;
    
    opts.max_tokens_per_batch = recommended_budget;
    opts.max_batch_size = 8; // reasonable default, will be limited by token budget anyway
    
    // Set hard cap only for extreme outliers (>p99)
    if (p99 > recommended_budget) {
        opts.hard_seq_len_cap = p99;
    }
    
    // Bucket step: finer granularity for more uniform data
    float cv = (p90 - p50) / std::max(1.0f, mean); // coefficient of variation proxy
    opts.bucket_step = cv > 0.5f ? 256 : 128;
    
    // Strategy based on data characteristics
    if (cv < 0.3f) {
        // Uniform lengths - similarity grouping works great
        opts.strategy = PackingStrategy::SIMILARITY_GROUPED;
        opts.similarity_threshold = 0.15f;
    } else if (cv < 0.6f) {
        // Moderate variance - best-fit works well
        opts.strategy = PackingStrategy::BEST_FIT_DECREASING;
    } else {
        // High variance - greedy is simpler and works fine
        opts.strategy = PackingStrategy::GREEDY;
    }
    
    // Gradient accumulation
    if (target_effective_batch > 0) {
        opts.target_effective_batch_size = target_effective_batch;
    }
    
    return opts;
}

// =============================================================================
// Estimate Batching (Quick Analysis)
// =============================================================================
BatchEstimate estimateBatching(const Catalog& catalog, const BatchOptions& opts) {
    BatchEstimate est{0, 0, 0.0f, opts.max_tokens_per_batch};
    
    const auto& entries = catalog.entries();
    if (entries.empty()) return est;
    
    std::vector<uint32_t> lengths;
    lengths.reserve(entries.size());
    for (const auto& e : entries) {
        if (e.seq_length > 0) lengths.push_back(e.seq_length);
    }
    if (lengths.empty()) return est;
    
    std::sort(lengths.begin(), lengths.end(), std::greater<uint32_t>());
    
    uint32_t budget = opts.max_tokens_per_batch;
    uint32_t max_size = opts.max_batch_size;
    uint64_t total_actual = 0;
    uint64_t total_compute = 0;
    
    // Simple first-fit estimate
    uint32_t current_max = 0;
    uint32_t current_count = 0;
    
    for (uint32_t len : lengths) {
        total_actual += len;
        
        // Check overflow
        if (len > budget) {
            est.estimated_overflow++;
            est.estimated_batches++;
            total_compute += len;
            continue;
        }
        
        uint32_t new_max = std::max(current_max, len);
        uint32_t new_count = current_count + 1;
        
        if (new_count > max_size || new_max * new_count > budget) {
            // Finalize current batch
            if (current_count > 0) {
                total_compute += current_max * current_count;
                est.estimated_batches++;
            }
            current_max = len;
            current_count = 1;
        } else {
            current_max = new_max;
            current_count = new_count;
        }
    }
    
    if (current_count > 0) {
        total_compute += current_max * current_count;
        est.estimated_batches++;
    }
    
    est.estimated_efficiency = total_compute > 0 
        ? static_cast<float>(total_actual) / total_compute 
        : 0.0f;
    
    // Recommend budget if efficiency is poor
    if (est.estimated_efficiency < opts.target_packing_efficiency) {
        // Try doubling budget
        BatchOptions test_opts = opts;
        test_opts.max_tokens_per_batch = budget * 2;
        BatchEstimate test_est = estimateBatching(catalog, test_opts);
        if (test_est.estimated_efficiency > est.estimated_efficiency + 0.05f) {
            est.recommended_token_budget = test_opts.max_tokens_per_batch;
        }
    }
    
    return est;
}

// =============================================================================
// Build Batches - Main Implementation
// =============================================================================
BatchSchedule buildBatches(const Catalog& catalog, const BatchOptions& opts) {
    BatchSchedule schedule{};
    
    if (catalog.entries().empty() || opts.max_tokens_per_batch == 0 || opts.max_batch_size == 0) {
        return schedule;
    }

    const auto& entries = catalog.entries();
    
    // Build sequence views with all metadata
    std::vector<SequenceView> views;
    views.reserve(entries.size());
    const uint32_t bucket_step = std::max(1u, opts.bucket_step);
    
    std::vector<uint32_t> all_lengths;
    all_lengths.reserve(entries.size());

    for (const auto& entry : entries) {
        if (entry.seq_length == 0) continue;
        
        views.push_back({
            entry.seq_id, 
            entry.seq_length, 
            bucketize(entry.seq_length, bucket_step),
            0.0f  // sort_key computed below
        });
        all_lengths.push_back(entry.seq_length);
    }

    if (views.empty()) return schedule;
    
    // Compute length percentiles for schedule stats
    std::sort(all_lengths.begin(), all_lengths.end());
    schedule.p50_seq_len = percentile(all_lengths, 0.50f);
    schedule.p90_seq_len = percentile(all_lengths, 0.90f);
    schedule.p99_seq_len = percentile(all_lengths, 0.99f);
    
    // Compute sort keys based on strategy
    for (auto& v : views) {
        float len_key = opts.prefer_short_first 
            ? static_cast<float>(v.length) 
            : -static_cast<float>(v.length);
        
        v.sort_key = len_key;
        
        // Apply curriculum: filter long sequences early in training
        if (opts.curriculum_progress < 1.0f) {
            uint32_t max_allowed = static_cast<uint32_t>(
                schedule.p50_seq_len + opts.curriculum_progress * (schedule.p99_seq_len - schedule.p50_seq_len)
            );
            if (v.length > max_allowed) {
                v.sort_key += 1e6f;  // push to end
            }
        }
    }
    
    // Sort based on strategy
    if (opts.strategy == PackingStrategy::SIMILARITY_GROUPED) {
        // Sort by bucket (groups similar lengths together)
        std::stable_sort(views.begin(), views.end(), [&](const SequenceView& a, const SequenceView& b) {
            if (a.bucket_key != b.bucket_key) {
                return opts.prefer_short_first ? (a.bucket_key < b.bucket_key) : (a.bucket_key > b.bucket_key);
            }
            return a.sort_key < b.sort_key;
        });
    } else if (opts.strategy == PackingStrategy::BEST_FIT_DECREASING) {
        // Sort longest first (FFD algorithm)
        std::stable_sort(views.begin(), views.end(), [](const SequenceView& a, const SequenceView& b) {
            return a.length > b.length;
        });
    } else {
        // GREEDY / Gradient-balanced: SHUFFLE to mix short and long sequences!
        // Issue #90: Length-sorted batching caused mode collapse at max_seq_len boundary.
        // By shuffling, we expose ALL position ranges from batch 1, preventing the
        // boundary effect where positions 671-1023 were only first seen in batch 5.
        if (opts.rng_seed != 0) {
            // Deterministic shuffle using provided seed
            std::mt19937_64 shuffle_rng(opts.rng_seed);
            std::shuffle(views.begin(), views.end(), shuffle_rng);
        } else {
            // Non-deterministic shuffle (fallback)
            std::random_device rd;
            std::mt19937_64 shuffle_rng(rd());
            std::shuffle(views.begin(), views.end(), shuffle_rng);
        }
    }
    
    // Determine effective token budget
    uint32_t token_budget = opts.max_tokens_per_batch;
    if (opts.adaptive_token_budget) {
        // Start with p75 * 3 as baseline, adjust based on target efficiency
        uint32_t adaptive_budget = percentile(all_lengths, 0.75f) * 3;
        adaptive_budget = std::max(adaptive_budget, opts.min_tokens_per_batch);
        token_budget = std::max(token_budget, adaptive_budget);
    }
    
    const uint32_t hard_cap = opts.hard_seq_len_cap > 0 ? opts.hard_seq_len_cap : UINT32_MAX;
    
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
        
        for (const auto& view : views) {
            const uint32_t seq_len = view.length;
            
            // Check for hard-cap overflow
            if (seq_len > token_budget || seq_len > hard_cap) {
                BatchAssignment overflow{};
                overflow.seq_ids.push_back(view.seq_id);
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
                uint32_t new_total = new_max * new_size;
                
                if (new_total > token_budget || new_size > opts.max_batch_size) continue;
                
                // Check similarity constraint if enabled
                if (opts.similarity_threshold < 1.0f) {
                    if (!areSimilarLengths(ob.assignment.min_seq_len, seq_len, opts.similarity_threshold) ||
                        !areSimilarLengths(ob.assignment.max_seq_len, seq_len, opts.similarity_threshold)) {
                        continue;
                    }
                }
                
                uint32_t old_total = ob.assignment.total_tokens;
                uint32_t waste_increase = new_total - old_total - seq_len;
                
                if (waste_increase < best_waste_increase) {
                    best_waste_increase = waste_increase;
                    best_batch = static_cast<int>(i);
                }
            }
            
            if (best_batch >= 0) {
                // Add to existing batch
                auto& ob = open_batches[best_batch];
                ob.assignment.seq_ids.push_back(view.seq_id);
                ob.assignment.max_seq_len = std::max(ob.assignment.max_seq_len, seq_len);
                ob.assignment.min_seq_len = std::min(ob.assignment.min_seq_len, seq_len);
                ob.assignment.actual_tokens += seq_len;
                ob.assignment.total_tokens = ob.assignment.max_seq_len * static_cast<uint32_t>(ob.assignment.seq_ids.size());
                ob.lengths.push_back(seq_len);
                ob.remaining_slots--;
                
                // Check if batch is now "full enough" to close
                float efficiency = static_cast<float>(ob.assignment.actual_tokens) / ob.assignment.total_tokens;
                if (ob.remaining_slots == 0 || efficiency >= 0.9f || 
                    ob.assignment.total_tokens >= token_budget * 0.85f) {
                    finalizeBatchStats(ob.assignment, ob.lengths);
                    schedule.batches.push_back(std::move(ob.assignment));
                    open_batches.erase(open_batches.begin() + best_batch);
                }
            } else {
                // Start new batch
                OpenBatch new_batch;
                new_batch.assignment.seq_ids.push_back(view.seq_id);
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
        // GREEDY / SIMILARITY_GROUPED / GRADIENT_BALANCED
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
        
        for (const auto& view : views) {
            const uint32_t seq_len = view.length;
            
            // Check for hard-cap overflow
            if (seq_len > token_budget || seq_len > hard_cap) {
                finalizeCurrent();
                BatchAssignment overflow{};
                overflow.seq_ids.push_back(view.seq_id);
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
            uint32_t prospective_tokens = prospective_size * prospective_max;
            
            bool exceeds_budget = prospective_tokens > token_budget;
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
            current.seq_ids.push_back(view.seq_id);
            current.max_seq_len = std::max(current.max_seq_len, seq_len);
            current.min_seq_len = current.min_seq_len == 0 ? seq_len : std::min(current.min_seq_len, seq_len);
            current.actual_tokens += seq_len;
            current.total_tokens = current.max_seq_len * static_cast<uint32_t>(current.seq_ids.size());
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
                // Fisher-Yates shuffle with deterministic seed for reproducibility
                std::mt19937 rng(42);
                for (size_t i = normal_batches.size() - 1; i > 0; --i) {
                    std::uniform_int_distribution<size_t> dist(0, i);
                    std::swap(normal_batches[i], normal_batches[dist(rng)]);
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
    
    // =======================================================================
    // Gradient accumulation grouping
    // =======================================================================
    
    if (opts.target_effective_batch_size > 0 && !schedule.batches.empty()) {
        // Group batches so each group has ~target sequences
        uint32_t target = opts.target_effective_batch_size;
        uint32_t group_id = 0;
        uint32_t group_seqs = 0;
        
        for (auto& batch : schedule.batches) {
            if (group_seqs >= target) {
                group_id++;
                group_seqs = 0;
            }
            batch.accumulation_group = group_id;
            group_seqs += static_cast<uint32_t>(batch.seq_ids.size());
        }
        
        schedule.num_accumulation_groups = group_id + 1;
        schedule.effective_batch_size = 
            static_cast<uint32_t>(batch_size_sum / std::max(1u, schedule.num_accumulation_groups));
    }
    
    return schedule;
}

} // namespace GRIM::Batching
