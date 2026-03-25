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
#include <memory>
#include <stdexcept>
#include <numeric>

#include "../TNC/Token-normalized_clipping.hpp"

// Forward declarations
struct TrainingSequence;

// Forward declaration — full definition in UnigramByte/AtomTable.hpp
namespace GRIM { namespace Tokenizer { class AtomTable; } }

namespace GRIM {

// Forward declaration — full definition in UnigramByte/UniByte.hpp
namespace Tokenizer { struct TokenLayout; }

namespace Batching {
struct BatchAssignment;
}

namespace Batching {

// =============================================================================
// TeacherStep — per-step ground truth for structured CE supervision
// =============================================================================
struct TeacherStep {
    int op_id;
    int arg1_slot;
    int arg2_slot;
    int write_slot;
    float expected_value;
};

// =============================================================================
// BatchPayload — immutable batch datum
// =============================================================================
struct BatchPayload {
    // ═══════════════════════════════════════════════════════════════════════════
    // IDENTITY (from BatchAssignment — carried through, not recomputed)
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<uint32_t> seq_ids;           // which sequences are in this batch

    // ═══════════════════════════════════════════════════════════════════════════
    // GEOMETRY (computed ONCE during buildBatchPayload)
    // Single source of truth for per-batch seq_len: downstream code must use
    // payload.batch_size, payload.max_seq_len, payload.total_tokens, etc. — not config.
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
    int min_seq_len = 0;                     // shortest sequence in batch (for logging)
    float length_variance = 0.0f;            // variance of seq_lengths (scheduler metric)
    bool overflow = false;                   // true if single seq exceeded token budget (scheduler)

    // ═══════════════════════════════════════════════════════════════════════════
    // PADDED DATA (flat [batch_size * max_seq_len] layout, computed ONCE)
    // ═══════════════════════════════════════════════════════════════════════════
    static constexpr int kTextFeatureDim = 16;

    std::vector<int> input_ids;              // [total_tokens] padded with 0
    std::vector<int> target_ids;             // [total_tokens] padded with -1
    std::vector<float> numeric_values;       // [total_tokens] padded with 0.0f
    std::vector<uint16_t> text_features;     // [total_tokens * kTextFeatureDim] padded with 0
    std::vector<uint8_t> atom_mask;          // [total_tokens] padded with 0 (1 = any atom type)
     std::vector<uint32_t> atom_flags;         // [total_tokens] padded with 0 (type-specific metadata from AtomTable)
    std::vector<int32_t> token_to_slot_map;   // [total_tokens] padded with -1 (slot_id for execution; -1 = non-state-bearing)

    // ═══════════════════════════════════════════════════════════════════════════
    // TEACHER EXECUTION STEPS (for structured CE supervision)
    // Populated for arithmetic batches; empty for non-execution batches.
    // teacher_steps[b][k] = TeacherStep for batch row b, execution step k.
    // When non-empty: teacher_steps.size() == batch_size, each inner vector
    // has exactly execution_block_num_steps entries (1:1 with ExecutionBlock steps).
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<std::vector<TeacherStep>> teacher_steps;

    // ═══════════════════════════════════════════════════════════════════════════
    // ATOM TABLE SIDE CHANNEL (host-only, NOT transferred to GPU)
    // atom_entry_ids[i] indexes into seq_atom_tables[batch_row] for token i.
    // kAtomEntryNone = no atom at this position.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<uint32_t> atom_entry_ids;    // [total_tokens] padded with kAtomEntryNone
    std::vector<std::shared_ptr<const GRIM::Tokenizer::AtomTable>> seq_atom_tables;  // [batch_size]

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

        // Cross-check: numeric and text feature arrays match total_tokens
        if (static_cast<int>(numeric_values.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.numeric_values.size()=" +
                std::to_string(numeric_values.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (static_cast<int>(atom_mask.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_mask.size()=" +
                std::to_string(atom_mask.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (static_cast<int>(atom_flags.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_flags.size()=" +
                std::to_string(atom_flags.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (static_cast<int>(token_to_slot_map.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.token_to_slot_map.size()=" +
                std::to_string(token_to_slot_map.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        const int expected_text_feat = total_tokens * kTextFeatureDim;
        if (static_cast<int>(text_features.size()) != expected_text_feat) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.text_features.size()=" +
                std::to_string(text_features.size()) + " != total_tokens*kTextFeatureDim=" +
                std::to_string(expected_text_feat));
        }

        // Teacher steps validation (when populated for arithmetic batches)
        if (!teacher_steps.empty()) {
            if (static_cast<int>(teacher_steps.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.teacher_steps.size()=" +
                    std::to_string(teacher_steps.size()) + " != batch_size=" +
                    std::to_string(batch_size));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // GPU TRANSFER GEOMETRY (precomputed byte sizes for cudaMemcpy callers)
    // ═══════════════════════════════════════════════════════════════════════════
    size_t inputIdBytes()      const { return static_cast<size_t>(total_tokens) * sizeof(int); }
    size_t targetIdBytes()     const { return static_cast<size_t>(total_tokens) * sizeof(int); }
    size_t numericValueBytes() const { return static_cast<size_t>(total_tokens) * sizeof(float); }
    size_t atomMaskBytes()     const { return static_cast<size_t>(total_tokens) * sizeof(uint8_t); }
    size_t atomFlagBytes()     const { return static_cast<size_t>(total_tokens) * sizeof(uint32_t); }
    size_t textFeatureBytes()  const { return static_cast<size_t>(total_tokens) * kTextFeatureDim * sizeof(uint16_t); }
    size_t slotMapBytes()      const { return static_cast<size_t>(total_tokens) * sizeof(int32_t); }
    size_t totalTransferBytes() const {
        return inputIdBytes() + targetIdBytes() + numericValueBytes() +
               atomMaskBytes() + atomFlagBytes() + textFeatureBytes() + slotMapBytes();
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
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t max_cached_batch,
    size_t max_cached_seq_len);

}  // namespace Batching
}  // namespace GRIM
