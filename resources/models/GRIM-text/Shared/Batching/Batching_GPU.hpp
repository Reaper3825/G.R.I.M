#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include "../DynaSeqs/DynaSeq_GPU.hpp"

namespace GRIM::Batching {

using GRIM::DynaSeq::Catalog;

// =============================================================================
// Packing Strategy
// =============================================================================
enum class PackingStrategy {
    GREEDY,              // Simple first-fit (fast, ~85% efficiency)
    BEST_FIT_DECREASING, // FFD bin-packing (slower, ~95% efficiency)
    SIMILARITY_GROUPED   // Group by length similarity (best for padding)
};

// =============================================================================
// Batch Ordering (post-packing sort)
// =============================================================================
enum class BatchOrdering {
    PRESERVE,            // Keep natural order from packing
    LENGTH_ASCENDING,    // Short batches first → long batches last (curriculum)
    LENGTH_DESCENDING,   // Long batches first → short batches last
    INTERLEAVED,         // Alternate short/long to smooth gradient variance
    RANDOM               // Shuffle batches randomly
};

// =============================================================================
// Batch Assignment (scheduler output)
// =============================================================================
// The scheduler (buildBatches) produces BatchAssignment: which seq_ids go in which
// batch and in what order. Downstream builds BatchPayload from each assignment
// (buildBatchPayload) and acts on the payload for training/validation.
struct BatchAssignment {
    std::vector<uint32_t> seq_ids;   // sequence ids in this batch
    uint32_t max_seq_len = 0;        // longest sequence length in the batch
    uint32_t min_seq_len = 0;        // shortest sequence length (for similarity metrics)
    uint32_t total_tokens = 0;       // max_seq_len * batch_size (compute cost)
    uint32_t padding_tokens = 0;     // waste = total_tokens - sum(actual lengths)
    uint32_t actual_tokens = 0;      // sum of real lengths
    float packing_efficiency = 0.0f; // actual_tokens / total_tokens
    float length_variance = 0.0f;    // variance of lengths in batch (lower = less padding)
    bool overflow = false;           // true if single seq exceeds hard cap (unavoidable)
};

// =============================================================================
// Batch Schedule with Statistics
// =============================================================================
struct BatchSchedule {
    std::vector<BatchAssignment> batches;
    
    // Token statistics
    uint64_t total_tokens = 0;           // total compute tokens (with padding)
    uint64_t actual_tokens = 0;          // actual content tokens
    uint64_t padding_tokens = 0;         // wasted tokens
    
    // Batch statistics
    uint32_t overflow_batches = 0;       // unavoidable single-seq batches
    uint32_t min_batch_size_observed = 0;
    uint32_t max_batch_size_observed = 0;
    uint32_t max_seq_len_observed = 0;
    float avg_batch_size = 0.0f;
    float avg_packing_efficiency = 0.0f; // mean efficiency across batches
    
    // Distribution info
    uint32_t p50_seq_len = 0;            // median sequence length
    uint32_t p90_seq_len = 0;            // 90th percentile
    uint32_t p99_seq_len = 0;            // 99th percentile
    
    // Get human-readable summary
    std::string summary() const;
};

// =============================================================================
// Batch Options
// =============================================================================
struct BatchOptions {
    // === Core limits ===
    uint32_t max_tokens_per_batch = 0;     // MUST be set by caller (Rule 20: no silent defaults)
    uint32_t max_batch_size = 32;          // max sequences per batch
    
    // === Packing strategy ===
    PackingStrategy strategy = PackingStrategy::SIMILARITY_GROUPED;
    uint32_t bucket_step = 128;            // length bucket granularity (smaller = better packing)
    float similarity_threshold = 0.25f;    // max length ratio variance in batch (0.25 = within 25%)
    
    // === Curriculum learning ===
    bool prefer_short_first = false;       // process shorter sequences first
    float curriculum_progress = 1.0f;      // 0.0 = start (short only), 1.0 = full curriculum
    
    // === Batch ordering (post-packing) ===
    BatchOrdering batch_ordering = BatchOrdering::LENGTH_ASCENDING;  // curriculum by default
    bool interleave_overflow = true;       // spread overflow batches throughout instead of clustering
    
    // === RNG for shuffling (Issue #90) ===
    uint64_t rng_seed = 0;                 // 0 = use random_device, >0 = deterministic shuffle
};

// =============================================================================
// Public API
// =============================================================================

// Build batches from catalog with given options
BatchSchedule buildBatches(const Catalog& catalog, const BatchOptions& opts);

} // namespace GRIM::Batching
