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
    RANDOM               // Shuffle fixed-size batches randomly
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
    
    // Token statistics. `sequence_cap` and `batch_size` are authored by the
    // startup config/hyperparameter path. Payloads pad each row to this
    // fixed cap, so schedule totals use batch_size * sequence_cap, not per-batch
    // observed max length. The scheduler must not derive geometry from a token
    // budget.
    uint32_t sequence_cap = 0;
    uint64_t total_tokens = 0;           // total compute tokens (with fixed padding)
    uint64_t actual_tokens = 0;          // actual content tokens
    uint64_t padding_tokens = 0;         // wasted tokens
    
    // Batch statistics
    uint32_t batch_size = 0;              // fixed run batch rows
    uint32_t max_seq_len_observed = 0;   // longest actual post-window sequence
    float packing_efficiency = 0.0f;     // actual_tokens / total_tokens
    uint32_t discarded_tail_sequences = 0; // valid rows dropped because they did not fill a full fixed batch
    
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
// seq_id consumed later by buildBatchPayload(). fixed_sequence_cap and
// fixed_batch_size must be the configured hyperparameter-authored run geometry;
// buildBatches fails if any post-window sequence exceeds that fixed cap.
struct PackerPolicy;
BatchSchedule buildBatches(
    const std::vector<uint32_t>& sequence_lengths,
    uint32_t fixed_sequence_cap,
    uint32_t fixed_batch_size,
    const PackerPolicy& policy);

} // namespace GRIM::Batching
