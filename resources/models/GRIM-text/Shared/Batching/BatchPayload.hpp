//======================================================//
//  BatchPayload.hpp
//  Single source of truth for per-batch metadata
//
//  REPLACES: BatchPreparationResult, batch_prep_* staging
//  vectors in TrainingState, and scattered metadata
//  recomputation across 5+ callsites.
//
//  One struct carries everything a batch needs from
//  sequence extraction through loss computation.
//  Built ONCE by buildBatchPayload(), consumed by all
//  downstream functions as const reference.
//
//  Rule 20: No fallbacks. validate() crashes with
//  detailed error on ANY inconsistency.
//======================================================//

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>

#include "../TNC/Token-normalized_clipping.hpp"

// Forward declarations
struct TrainingSequence;

namespace GRIM {

namespace Batching {
struct BatchAssignment;
}

namespace Batching {

// =============================================================================
// BatchPayload — immutable batch datum
// =============================================================================
struct BatchPayload {
    // ═══════════════════════════════════════════════════════════════════════════
    // IDENTITY (from BatchAssignment — carried through, not recomputed)
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<uint32_t> seq_ids;           // which sequences are in this batch
    uint32_t accumulation_group = 0;         // gradient accumulation group

    // ═══════════════════════════════════════════════════════════════════════════
    // GEOMETRY (computed ONCE during buildBatchPayload)
    // ═══════════════════════════════════════════════════════════════════════════
    int batch_size = 0;                      // number of sequences
    int max_seq_len = 0;                     // longest sequence (pad target)
    int total_tokens = 0;                    // batch_size * max_seq_len (includes padding)
    int actual_tokens = 0;                   // sum of real sequence lengths (no padding)
    int padding_tokens = 0;                  // total_tokens - actual_tokens
    int valid_tokens = 0;                    // total unmasked targets (for loss mean reduction)
    std::vector<int> seq_lengths;            // [batch_size] — original length per sequence before padding
    std::vector<int> valid_target_counts;    // [batch_size] — unmasked targets per sequence
    float packing_efficiency = 0.0f;         // actual_tokens / total_tokens

    // ═══════════════════════════════════════════════════════════════════════════
    // PADDED DATA (flat [batch_size * max_seq_len] layout, computed ONCE)
    // ═══════════════════════════════════════════════════════════════════════════
    static constexpr int kTextFeatureDim = 16;

    std::vector<int> input_ids;              // [total_tokens] padded with 0
    std::vector<int> target_ids;             // [total_tokens] padded with -1
    std::vector<float> numeric_values;       // [total_tokens] padded with 0.0f
    std::vector<uint8_t> numeric_mask;       // [total_tokens] padded with 0
    std::vector<uint16_t> text_features;     // [total_tokens * kTextFeatureDim] padded with 0
    std::vector<uint8_t> text_mask;          // [total_tokens] padded with 0

    // ═══════════════════════════════════════════════════════════════════════════
    // TOKEN STATS (for gradient clipping — computed ONCE)
    // ═══════════════════════════════════════════════════════════════════════════
    TNC::BatchTokenStats token_stats;

    // ═══════════════════════════════════════════════════════════════════════════
    // CACHE FIT (computed ONCE against model limits)
    // ═══════════════════════════════════════════════════════════════════════════
    bool fits_in_cache = false;

    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION (Rule 20: Crash with detailed error)
    // ═══════════════════════════════════════════════════════════════════════════
    void validate(const char* caller) const {
        if (batch_size <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.batch_size=" +
                std::to_string(batch_size) + " (must be > 0)");
        }
        if (max_seq_len <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.max_seq_len=" +
                std::to_string(max_seq_len) + " (must be > 0)");
        }
        if (total_tokens != batch_size * max_seq_len) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.total_tokens=" +
                std::to_string(total_tokens) + " != batch_size(" +
                std::to_string(batch_size) + ") * max_seq_len(" +
                std::to_string(max_seq_len) + ")=" +
                std::to_string(batch_size * max_seq_len));
        }
        if (valid_tokens <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.valid_tokens=" +
                std::to_string(valid_tokens) + " (must be > 0 — batch has no trainable targets)");
        }
        if (static_cast<int>(seq_lengths.size()) != batch_size) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.seq_lengths.size()=" +
                std::to_string(seq_lengths.size()) + " != batch_size=" +
                std::to_string(batch_size));
        }
        if (static_cast<int>(valid_target_counts.size()) != batch_size) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.valid_target_counts.size()=" +
                std::to_string(valid_target_counts.size()) + " != batch_size=" +
                std::to_string(batch_size));
        }
        if (static_cast<int>(input_ids.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.input_ids.size()=" +
                std::to_string(input_ids.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (static_cast<int>(target_ids.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.target_ids.size()=" +
                std::to_string(target_ids.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        // Cross-check: sum of valid_target_counts must equal valid_tokens
        const int vtc_sum = std::accumulate(
            valid_target_counts.begin(), valid_target_counts.end(), 0);
        if (vtc_sum != valid_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.valid_tokens=" +
                std::to_string(valid_tokens) + " != sum(valid_target_counts)=" +
                std::to_string(vtc_sum));
        }
        // Cross-check: sum of seq_lengths must equal actual_tokens
        const int sl_sum = std::accumulate(
            seq_lengths.begin(), seq_lengths.end(), 0);
        if (sl_sum != actual_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.actual_tokens=" +
                std::to_string(actual_tokens) + " != sum(seq_lengths)=" +
                std::to_string(sl_sum));
        }
        if (!fits_in_cache) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.fits_in_cache=false — batch exceeds GPU cache limits");
        }
    }
};

// =============================================================================
// Builder — constructs a validated BatchPayload from raw components
// =============================================================================

/**
 * Build a complete, validated, padded batch payload.
 *
 * This is the SINGLE point where:
 *   - Sequences are extracted from views
 *   - max_seq_len is computed
 *   - Padding is applied
 *   - Targets are validated and final-position-masked
 *   - valid_tokens is counted
 *   - Token stats for clipping are computed
 *   - Cache fit is checked
 *
 * @param assignment     Batch assignment from scheduler (seq_ids, etc.)
 * @param views          Sequence data views indexed by seq_id
 * @param vocab_size     Actual vocab size for target validation
 * @param max_cached_batch   GPU cache batch capacity
 * @param max_cached_seq_len GPU cache sequence length capacity
 * @return Complete BatchPayload ready for downstream consumption
 */
BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<TrainingSequence*>& views,
    int vocab_size,
    size_t max_cached_batch,
    size_t max_cached_seq_len);

}  // namespace Batching
}  // namespace GRIM
