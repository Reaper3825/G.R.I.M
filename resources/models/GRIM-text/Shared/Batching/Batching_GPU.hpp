#pragma once
#include <vector>
#include <cstdint>
#include <string>

namespace GRIM::Batching {

// =============================================================================
// Batch Ordering
// =============================================================================
enum class BatchOrdering {
    PRESERVE,            // Keep natural order from packing
    RANDOM               // Shuffle completed batches randomly
};

// =============================================================================
// Batch Assignment (scheduler output)
// =============================================================================
// The scheduler (buildBatches) produces BatchAssignment: which seq_ids go in which
// batch and in what order. It does NOT author sequence geometry. Downstream
// buildBatchPayload() reads the selected sequences, validates the sliding-window
// cap, and pads every row to the fixed run sequence cap.
struct BatchAssignment {
    std::vector<uint32_t> seq_ids;   // sequence ids in this batch
};

// =============================================================================
// Batch Schedule with Statistics
// =============================================================================
struct BatchSchedule {
    std::vector<BatchAssignment> batches;
    
    // Schedule-level token statistics. `actual_tokens` is true content-token
    // count across emitted batches. The remaining token fields are
    // projections derived from the caller-authored fixed rectangle
    // (selected rows * fixed_sequence_cap) that BatchPayload
    // materializes later; the scheduler itself does not own or mirror that
    // config geometry.
    uint64_t projected_total_tokens = 0;   // projected fixed-rectangle compute tokens
    uint64_t actual_tokens = 0;            // actual content tokens across emitted batches
    uint64_t projected_padding_tokens = 0; // projected fixed-rectangle padding tokens
    
    // Batch statistics
    uint32_t batch_size = 0;              // maximum rows fitting the token budget
    uint32_t token_budget = 0;            // padded token slots allowed per batch
    uint32_t max_seq_len_observed = 0;   // longest actual post-window sequence
    float projected_packing_efficiency = 0.0f; // actual_tokens / projected_total_tokens
    uint32_t discarded_tail_sequences = 0; // compatibility statistic; tails are emitted
    
    // Distribution info
    uint32_t p50_seq_len = 0;            // median sequence length
    uint32_t p90_seq_len = 0;            // 90th percentile
    uint32_t p99_seq_len = 0;            // 99th percentile
    
    // Get human-readable summary
    std::string summary() const;
};

// =============================================================================
// Public API
// =============================================================================

// Build batches from sequence lengths with given options. The vector index is the
// seq_id consumed later by buildBatchPayload(). Context width stays fixed;
// token_budget limits the padded rectangle without changing row boundaries.
// buildBatches fails if any post-window sequence exceeds the context cap.
struct PackerPolicy;
BatchSchedule buildBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t fixed_sequence_cap,
    uint32_t token_budget,
    const PackerPolicy& policy);

} // namespace GRIM::Batching
