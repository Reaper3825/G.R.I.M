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
#include "../Execution/ExecutionMetadata.hpp"

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

// TeacherStep is now defined in Execution/ExecutionMetadata.hpp as GRIM::Execution::TeacherStep.
// This alias preserves call-site compatibility during the cutover.
using TeacherStep = GRIM::Execution::TeacherStep;

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
    // LM-supervised token count AFTER execution-slot target masking.
    // Set by buildBatchPayload() Phase 4b.  Equals valid_tokens when no
    // execution-slot masking occurs.
    int lm_valid_tokens = 0;
    int vocab_size = 0;                      // vocabulary size (for loss kernels + target validation)
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

    // Device mirror of token_to_slot_map.  Non-owning — points into the GPU
    // cache allocated by TrainingState (training) or the inference upload buffer.
    // Set after cudaMemcpyAsync; read-only by execution kernels.
    // mutable: payload is passed as const& but device ptr is set post-upload.
    mutable const int32_t* d_token_to_slot_map = nullptr;

    // ═══════════════════════════════════════════════════════════════════════════
    // TEACHER EXECUTION STEPS (for structured CE supervision)
    // Populated for arithmetic batches; empty for non-execution batches.
    // teacher_steps[b][k] = TeacherStep for batch row b, execution step k.
    // When non-empty: teacher_steps.size() == batch_size, each inner vector
    // has exactly execution_block_num_steps entries (1:1 with ExecutionBlock steps).
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<std::vector<TeacherStep>> teacher_steps;

    // ═══════════════════════════════════════════════════════════════════════════
    // TEACHER STEP MASK (for padded-step loss zeroing)
    // teacher_step_mask[b][k] = 1 for real steps, 0 for padded steps.
    // Constructed at batch-build time from original teacher_steps count vs
    // execution_block_num_steps. Loss loops skip steps where mask == 0.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<std::vector<uint8_t>> teacher_step_mask;

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPILED STRUCTURED-EXECUTION PAYLOAD
    //
    // execution_active[b] is the AUTHORITATIVE activation bit per row.
    // Non-empty teacher_steps is a supervised-training payload validity
    // requirement, NOT the activation source.
    //
    // token_to_slot_map (above) is the runtime binding projection.
    // teacher_steps (above) is the supervision projection.
    // compiled_bootstrap_bindings is the compiled provenance.
    // slot_selection_targets is per-row dense selector supervision.
    //
    // Runtime D_row is reconstructed from compiled_bootstrap_bindings ∪ teacher_steps.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<bool> execution_active;    // [batch_size] — authoritative per-row activation
    std::vector<std::vector<GRIM::Execution::CompiledBootstrapBinding>> compiled_bootstrap_bindings;  // [batch_size]
    std::vector<std::vector<GRIM::Execution::SlotSelectionTarget>> slot_selection_targets;  // [batch_size]

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
        if (vocab_size <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.vocab_size=" +
                std::to_string(vocab_size) + " (must be > 0)");
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

        // Compiled execution payload validation
        if (!execution_active.empty()) {
            if (static_cast<int>(execution_active.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.execution_active.size()=" +
                    std::to_string(execution_active.size()) + " != batch_size=" +
                    std::to_string(batch_size));
            }
            // execution_active with false + non-empty teacher_steps is a structural error
            for (int b = 0; b < batch_size; ++b) {
                if (!execution_active[b] && !teacher_steps.empty()
                    && !teacher_steps[b].empty()) {
                    throw std::runtime_error(
                        std::string(caller) + ": row " + std::to_string(b) +
                        " has execution_active=false but non-empty teacher_steps — "
                        "teacher_steps alone do not activate execution");
                }
            }
        }
        if (!compiled_bootstrap_bindings.empty()) {
            if (static_cast<int>(compiled_bootstrap_bindings.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.compiled_bootstrap_bindings.size()=" +
                    std::to_string(compiled_bootstrap_bindings.size()) + " != batch_size=" +
                    std::to_string(batch_size));
            }
        }
        if (!slot_selection_targets.empty()) {
            if (static_cast<int>(slot_selection_targets.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.slot_selection_targets.size()=" +
                    std::to_string(slot_selection_targets.size()) + " != batch_size=" +
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
 * @param execution_num_slots  ExecutionBlock slot count (from config)
 * @param execution_num_ops    ExecutionBlock op count (from config)
 * @param execution_num_steps  ExecutionBlock step count (from config)
 * @return Complete BatchPayload ready for downstream consumption
 */
BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<TrainingSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t max_cached_batch,
    size_t max_cached_seq_len,
    int execution_num_slots,
    int execution_num_ops,
    int execution_num_steps);

}  // namespace Batching
}  // namespace GRIM
