//======================================================//
//  BatchPayload.hpp
//  Realized per-batch runtime payload
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

#include "../Execution/ExecutionMetadata.hpp"

// Forward declaration for GRMT-authored training rows.
namespace GRIM { namespace TokenizerArtifacts { struct GrmtSequence; } }

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

enum class BatchPayloadMode {
    Training,
    InferencePrefill,
    InferenceDecode
};

// =============================================================================
// BatchPayload — immutable batch datum
// =============================================================================
struct BatchPayload {
    BatchPayloadMode mode = BatchPayloadMode::Training;

    // ═══════════════════════════════════════════════════════════════════════════
    // IDENTITY (from BatchAssignment — carried through, not recomputed)
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<uint32_t> seq_ids;           // which sequences are in this batch

    // ═══════════════════════════════════════════════════════════════════════════
    // GEOMETRY (computed/materialized ONCE during buildBatchPayload)
    // For fixed-shape training/eval, batch_size/max_seq_len are config-owned
    // runtime facts authored through HyperParameters/HyperparameterGroupings
    // and realized here for row layout / payload validation. Inference
    // prefill/decode payloads carry their explicit runtime geometry.
    // ═══════════════════════════════════════════════════════════════════════════
    int batch_size = 0;                      // training: fixed batch rows; inference: prompt/decode rows
    int max_seq_len = 0;                     // training: fixed sequence cap; inference: prompt/decode length
    int total_tokens = 0;                    // batch_size * max_seq_len (includes padding)
    int actual_tokens = 0;                   // sum of real sequence lengths (no padding); telemetry/diagnostics token count owner
    int padding_tokens = 0;                  // total_tokens - actual_tokens
    int valid_tokens = 0;                    // total unmasked targets (for loss mean reduction)
    // LM-supervised token count AFTER execution-slot target masking.
    // Set by buildBatchPayload() Phase 4b.  Equals valid_tokens when no
    // execution-slot masking occurs.
    int lm_valid_tokens = 0;
    int vocab_size = 0;                      // vocabulary size (for loss kernels + target validation)
    std::vector<int> seq_lengths;            // [batch_size] — original length per sequence before padding
    std::vector<int> valid_target_counts;    // [batch_size] — unmasked targets per sequence

    // ═══════════════════════════════════════════════════════════════════════════
    // PADDED DATA (flat [batch_size * max_seq_len] layout, computed ONCE)
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<int> input_ids;              // [total_tokens] padded with Tokenizer::PAD_TOKEN_ID
    std::vector<int> target_ids;             // [total_tokens] padded with -1
    std::vector<float> numeric_values;       // [total_tokens] padded with 0.0f
    std::vector<uint8_t> atom_mask;          // [total_tokens] padded with 0 (1 = any atom type)
    std::vector<uint32_t> atom_flags;         // [total_tokens] padded with 0 (type-specific metadata from AtomTable)
    std::vector<int32_t> token_to_slot_map;   // [total_tokens] padded with -1 (slot_id for execution; -1 = non-state-bearing)

    // NOTE: Device pointers used to live here as `mutable d_token_to_slot_map`
    // and `mutable d_atom_mask`, written by the upload path and read by the
    // forward/loss path. They have moved to `GRIM::Batching::BatchDeviceBindings`
    // (Shared/Batching/BatchDeviceBindings.hpp). BatchPayload is host-only and
    // immutable after buildBatchPayload() returns — see the header banner and
    // the BatchPayload contract section in
    // .cursor/plans/precomputebatchpayloads_*.plan.md.

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
    // MTP (Multi-Token Prediction) SHIFTED TARGETS
    //
    // Computed ONCE during buildBatchPayload() when mtp_k > 0.
    // mtp_shifted_targets[k] = [total_tokens] shifted target_ids for head k.
    // mtp_valid_counts[k] = number of valid (non -1) targets after shift.
    //
    // Shift rule: for head k, shift = k+1.
    //   shifted[b*S + t] = target_ids[b*S + t + shift]  if (t + shift) < seq_len[b]
    //                     = -1                            otherwise
    //
    // Execution-slot masking is inherited: target_ids are already masked by
    // Phase 4b before MTP shift, so shifted targets respect slot boundaries.
    //
    // These are the AUTHORITATIVE shifted targets. MTP_GPU must use these
    // directly — no ad-hoc GPU shifting.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<std::vector<int>> mtp_shifted_targets;  // [K][total_tokens]
    std::vector<int> mtp_valid_counts;                   // [K] valid targets per head

    // ═══════════════════════════════════════════════════════════════════════════
    // CACHE FIT (computed ONCE against model limits)
    // ═══════════════════════════════════════════════════════════════════════════
    bool fits_in_cache = false;

    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION (Rule 20: Crash with detailed error)
    // ═══════════════════════════════════════════════════════════════════════════
    bool isTraining() const { return mode == BatchPayloadMode::Training; }
    bool isInference() const { return mode != BatchPayloadMode::Training; }
    bool isInferencePrefill() const { return mode == BatchPayloadMode::InferencePrefill; }
    bool isInferenceDecode() const { return mode == BatchPayloadMode::InferenceDecode; }
    bool ownsHostInputData() const {
        return mode == BatchPayloadMode::Training || mode == BatchPayloadMode::InferencePrefill;
    }
    bool hasTrainingTargets() const { return mode == BatchPayloadMode::Training; }

    const char* modeName() const {
        switch (mode) {
            case BatchPayloadMode::Training: return "training";
            case BatchPayloadMode::InferencePrefill: return "inference_prefill";
            case BatchPayloadMode::InferenceDecode: return "inference_decode";
        }
        throw std::runtime_error("BatchPayload.mode contains an unknown value");
    }

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
        if (isTraining() && valid_tokens <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.valid_tokens=" +
                std::to_string(valid_tokens) + " (must be > 0 — batch has no trainable targets)");
        }
        if (isTraining() && lm_valid_tokens <= 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.lm_valid_tokens=" +
                std::to_string(lm_valid_tokens) +
                " (must be > 0 after execution-slot target masking)");
        }
        if (isInference() && valid_tokens != 0) {
            throw std::runtime_error(
                std::string(caller) + ": inference BatchPayload.valid_tokens=" +
                std::to_string(valid_tokens) + " (must be 0; inference carries no training targets)");
        }
        if (isInference() && lm_valid_tokens != 0) {
            throw std::runtime_error(
                std::string(caller) + ": inference BatchPayload.lm_valid_tokens=" +
                std::to_string(lm_valid_tokens) + " (must be 0; inference carries no LM loss targets)");
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
        if (isTraining() && static_cast<int>(valid_target_counts.size()) != batch_size) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.valid_target_counts.size()=" +
                std::to_string(valid_target_counts.size()) + " != batch_size=" +
                std::to_string(batch_size));
        }
        if (isInference() && !valid_target_counts.empty()) {
            if (static_cast<int>(valid_target_counts.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": inference BatchPayload.valid_target_counts.size()=" +
                    std::to_string(valid_target_counts.size()) + " != batch_size=" +
                    std::to_string(batch_size));
            }
            for (int b = 0; b < batch_size; ++b) {
                if (valid_target_counts[b] != 0) {
                    throw std::runtime_error(
                        std::string(caller) + ": inference BatchPayload.valid_target_counts[" +
                        std::to_string(b) + "]=" + std::to_string(valid_target_counts[b]) +
                        " (must be 0; inference carries no training targets)");
                }
            }
        }
        if (ownsHostInputData() && static_cast<int>(input_ids.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.input_ids.size()=" +
                std::to_string(input_ids.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (hasTrainingTargets() && static_cast<int>(target_ids.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.target_ids.size()=" +
                std::to_string(target_ids.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (isInference() && !target_ids.empty()) {
            if (static_cast<int>(target_ids.size()) != total_tokens) {
                throw std::runtime_error(
                    std::string(caller) + ": inference BatchPayload.target_ids.size()=" +
                    std::to_string(target_ids.size()) + " != total_tokens=" +
                    std::to_string(total_tokens));
            }
            for (int i = 0; i < total_tokens; ++i) {
                if (target_ids[i] != -1) {
                    throw std::runtime_error(
                        std::string(caller) + ": inference BatchPayload.target_ids[" +
                        std::to_string(i) + "]=" + std::to_string(target_ids[i]) +
                        " (must be -1; inference must not smuggle supervision)");
                }
            }
        }
        // Cross-check: sum of valid_target_counts must equal valid_tokens
        if (isTraining()) {
            const int vtc_sum = std::accumulate(
                valid_target_counts.begin(), valid_target_counts.end(), 0);
            if (vtc_sum != valid_tokens) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.valid_tokens=" +
                    std::to_string(valid_tokens) + " != sum(valid_target_counts)=" +
                    std::to_string(vtc_sum));
            }
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

        // Cross-check: per-token atom metadata arrays match total_tokens
        if (ownsHostInputData() && static_cast<int>(numeric_values.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.numeric_values.size()=" +
                std::to_string(numeric_values.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (ownsHostInputData() && static_cast<int>(atom_mask.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_mask.size()=" +
                std::to_string(atom_mask.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (ownsHostInputData() && static_cast<int>(atom_flags.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_flags.size()=" +
                std::to_string(atom_flags.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (ownsHostInputData() && static_cast<int>(token_to_slot_map.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.token_to_slot_map.size()=" +
                std::to_string(token_to_slot_map.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
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

        // MTP shifted targets validation
        const int mtp_k = static_cast<int>(mtp_shifted_targets.size());
        if (mtp_k > 0) {
            if (static_cast<int>(mtp_valid_counts.size()) != mtp_k) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload.mtp_valid_counts.size()=" +
                    std::to_string(mtp_valid_counts.size()) + " != mtp_shifted_targets.size()=" +
                    std::to_string(mtp_k));
            }
            for (int k = 0; k < mtp_k; ++k) {
                if (static_cast<int>(mtp_shifted_targets[k].size()) != total_tokens) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload.mtp_shifted_targets[" +
                        std::to_string(k) + "].size()=" +
                        std::to_string(mtp_shifted_targets[k].size()) +
                        " != total_tokens=" + std::to_string(total_tokens));
                }
                if (mtp_valid_counts[k] < 0) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload.mtp_valid_counts[" +
                        std::to_string(k) + "]=" + std::to_string(mtp_valid_counts[k]) +
                        " — negative count is impossible (data corruption)");
                }
                // mtp_valid_counts[k] == 0 is LEGAL: short / heavily-masked
                // batches may have valid LM targets but no valid targets at
                // horizon (k+1). MTP_GPU treats zero-count heads as zero
                // contribution (no loss, no grad). Aborting the batch would
                // discard otherwise-valid LM training signal.
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
    size_t slotMapBytes()      const { return static_cast<size_t>(total_tokens) * sizeof(int32_t); }
    size_t totalTransferBytes() const {
        return inputIdBytes() + targetIdBytes() + numericValueBytes() +
               atomMaskBytes() + atomFlagBytes() + slotMapBytes();
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
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t max_cached_batch,
    size_t max_cached_seq_len,
    int execution_num_slots,
    int execution_num_ops,
    int execution_num_steps,
    int mtp_k);

/**
 * Build a validated single-row inference prefill payload from tokenizer-authored
 * metadata. This is the inference data-ingestion boundary: callers provide all
 * per-token atom side channels explicitly, and downstream CUDA code consumes the
 * resulting BatchPayload + BatchDeviceBindings pair.
 */
BatchPayload buildInferenceBatchPayload(
    const std::vector<int>& token_ids,
    const std::vector<float>& numeric_values,
    const std::vector<uint8_t>& atom_mask,
    const std::vector<uint32_t>& atom_flags,
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table,
    const std::vector<uint32_t>& atom_entry_ids,
    const std::vector<int32_t>& token_to_slot_map,
    int vocab_size,
    size_t max_cached_batch,
    size_t max_cached_seq_len,
    int execution_num_slots);

/** Build a single-token inference decode geometry payload. */
BatchPayload buildInferenceDecodePayload(int vocab_size);

}  // namespace Batching
}  // namespace GRIM
