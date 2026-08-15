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

#include "../Goal/GoalSpanView.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric>

// Forward declaration for GRMT-authored training rows.
namespace GRIM {
struct Goal;
namespace TokenizerArtifacts { struct GrmtSequence; }
}

// Forward declaration — full definition in UnigramByte/AtomTable.hpp
namespace GRIM { namespace Tokenizer { class AtomTable; } }

namespace GRIM {

// Forward declaration — full definition in UnigramByte/UniByte.hpp
namespace Batching {
struct BatchAssignment;
struct BatchDeviceStorage;
}

namespace Batching {

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
    // LM-supervised token count. Training payloads keep this aligned with
    // valid_tokens; inference payloads carry no loss targets.
    int lm_valid_tokens = 0;
    int vocab_size = 0;                      // vocabulary size (for loss kernels + target validation)
    std::vector<int> seq_lengths;            // [batch_size] — original length per sequence before padding
    std::vector<int> valid_target_counts;    // [batch_size] — unmasked targets per sequence
    // Logical <prompt>...</prompt> boundaries. Delimiters never appear in
    // input_ids. For a non-empty span, start = end - length + 1.
    std::vector<int32_t> prompt_lengths;       // [batch_size], 0 = no complete prompt in this row
    std::vector<int32_t> prompt_end_positions; // [batch_size], inclusive; -1 when length is 0
    // Immutable authored Goal metadata aligned one-to-one with training rows.
    // Entries may be null when a source row has no goal identifier.
    std::vector<std::shared_ptr<const GRIM::Goal>> goals; // [batch_size]

    // ═══════════════════════════════════════════════════════════════════════════
    // PADDED DATA (flat [batch_size * max_seq_len] layout, computed ONCE)
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<int> input_ids;              // [total_tokens] padded with Tokenizer::PAD_TOKEN_ID
    std::vector<int> target_ids;             // [total_tokens] padded with -1
    std::vector<float> numeric_values;       // [total_tokens] padded with 0.0f
    std::vector<uint8_t> atom_mask;          // [total_tokens] padded with 0 (1 = atom opening metadata anchor)
    // Causal prediction-row ownership authored by sliding-window construction.
    // For <TYPE> value </TYPE>, 1 spans the opening boundary through the final
    // value-emission row, so the auxiliary head predicts digit/place emissions
    // followed by </TYPE>. The close-boundary row is 0; the LM resumes there.
    // Host-only at this boundary; device upload and target routing belong to
    // the auxiliary-head integration that follows this payload-population step.
    std::vector<uint8_t> atom_aux_target_mask; // [total_tokens] padded with 0
    std::vector<uint32_t> atom_flags;         // [total_tokens] padded with 0 (opening-only metadata from AtomTable)
    // Dense bootstrap slot addresses used to materialize authored ARG seeds.
    std::vector<int32_t> token_to_slot_index_map; // [total_tokens], -1 = non-state-bearing

    // Compact authored atom facts. Each position identifies a metadata-bearing
    // typed opening boundary, never span content or a closing boundary. These
    // are materialized ONCE behind the payload boundary and uploaded as-is for
    // ScratchBlock consumption.
    // They are semantic data, not forward-time workspace.
    std::vector<int> atom_positions;          // [num_atoms] flat token indices into input_ids/numeric_values/atom_flags
    std::vector<int> atom_types;              // [num_atoms] Tokenizer::AtomType enum values aligned with atom_positions

    // NOTE: Device pointers used to live here as `mutable d_token_to_slot_index_map`
    // and `mutable d_atom_mask`, written by the upload path and read by the
    // forward/loss path. They have moved to `GRIM::Batching::BatchDeviceBindings`
    // (Shared/Batching/BatchDeviceBindings.hpp). The underlying device buffer
    // ownership is explicit on this payload via `device_storage`; callers may
    // borrow step-local addresses only through `BatchDeviceBindings`.
    // Host semantic fields remain immutable after buildBatchPayload() returns.
    std::shared_ptr<BatchDeviceStorage> device_storage;

    // ═══════════════════════════════════════════════════════════════════════════
    // ATOM TABLE SIDE CHANNEL (host-only, NOT transferred to GPU)
    // atom_entry_ids[i] indexes into seq_atom_tables[batch_row] only when token
    // i is a typed atom opening boundary. kAtomEntryNone is required elsewhere.
    // ═══════════════════════════════════════════════════════════════════════════
    std::vector<uint32_t> atom_entry_ids;    // [total_tokens] padded with kAtomEntryNone
    std::vector<std::shared_ptr<const GRIM::Tokenizer::AtomTable>> seq_atom_tables;  // [batch_size]

    // NUMBER AUXILIARY TARGETS (compact, aligned with atom_positions)
    //
    // These are supervision, never NumberEncoder forward inputs. Numeric
    // openings carry their complete AtomEntry::arg_number decomposition;
    // non-numeric openings retain zero/default rows so atom_types remains the
    // single extensible typed-routing authority. Numeric decoder coordinates
    // are compact [atom, step]; token rows only determine whether an atom has
    // unmasked supervision in this window.
    //
    // Layout (A = authoredAtomCount(), S = number_aux_target_digit_slots):
    //   number_aux_target_* scalar channels [A]
    //   number_aux_target_digits            [A * S]
    //   number_aux_target_pow10_index        [A * S]
    //   number_aux_target_digit_mask         [A * S]
    // pow10_index maps signed place p to p + number_aux_target_max_abs_pow10.
    int number_aux_target_digit_slots = 0;
    int number_aux_target_max_abs_pow10 = 0;
    int number_aux_target_valid_count = 0;
    std::vector<uint8_t> number_aux_target_valid;
    std::vector<uint8_t> number_aux_target_sign_negative;
    std::vector<uint8_t> number_aux_target_base;
    std::vector<uint16_t> number_aux_target_digit_count;
    std::vector<int> number_aux_target_digits;
    std::vector<int> number_aux_target_pow10_index;
    std::vector<uint8_t> number_aux_target_digit_mask;
    // Exact decoded entry value used by the later normalized-value objective.
    std::vector<double> number_aux_target_float_values;
    std::vector<int64_t> number_aux_target_int_values;
    std::vector<uint8_t> number_aux_target_numeric_kinds;

    // ═══════════════════════════════════════════════════════════════════════════
    // NUMBER ENCODER DIGIT-PLACE CHANNELS (compact, aligned with atom_positions)
    //
    // Materialized ONCE during payload build from each atom's CURRENT-token
    // arg_number metadata. These are
    // input-side features for token t only — target-side (t+1) metadata is
    // supervision and is NEVER materialized into this input channel (hard
    // anti-leakage rule). Padding digit slots have atom_digit_mask == 0 and
    // MUST contribute exactly zero in any consumer (forward pool or loss).
    //
    // Layout (A = authoredAtomCount(), S = number_encoder_digit_slots):
    //   atom_digit_values        [A * S]   digit identity 0..9, pad 0
    //   atom_digit_pow10_index   [A * S]   pow10 bucket index 0..2*max_abs_pow10, pad 0
    //   atom_digit_mask          [A * S]   1.0 real slot / 0.0 pad slot
    //   atom_digit_slot_features [A * S * kNumberSlotFeatureDim]
    //   atom_global_features     [A * kNumberGlobalFeatureDim]
    // Empty (and S == 0) when the NumberEncoder is disabled.
    // ═══════════════════════════════════════════════════════════════════════════
    static constexpr int kNumberSlotFeatureDim = 5;    // [digit/9, pow10_norm, is_zero_digit, sign, is_float_atom]
    static constexpr int kNumberGlobalFeatureDim = 6;  // [sign, exp_norm, int_digits_norm, frac_digits_norm, has_decimal, is_float_atom]
    int number_encoder_digit_slots = 0;                // realized per-atom slot capacity (0 = disabled)
    std::vector<int> atom_digit_values;
    std::vector<int> atom_digit_pow10_index;
    std::vector<float> atom_digit_mask;
    std::vector<float> atom_digit_slot_features;
    std::vector<float> atom_global_features;

    // ═══════════════════════════════════════════════════════════════════════════
    // CANDIDATE ATOM-ENTRY POOL
    //
    // The "menu" of options the selector chooses among: every row's AtomTable
    // entries, merged into ONE batch-global pool so the device selector can score
    // candidates. Execution-INDEPENDENT (data only; never derived from
    // ExecutionMemory — see GRIM/Docs/DeletedCode.md). Row r's candidate window is
    // [row_atom_offset[r], row_atom_offset[r+1]); a token's own entry maps to the
    // batch-global pool index row_atom_offset[row] + <row-local atom_entry_id>.
    //
    // Materialized only when the selector is enabled. Candidate identity and
    // exact payload metadata are independent of NumberEncoder construction.
    // ═══════════════════════════════════════════════════════════════════════════
    int num_pool_atoms = 0;
    std::vector<int> row_atom_offset;          // [batch_size + 1] prefix offsets into the pool
    std::vector<float> pool_numeric_values;    // [E] float view (matching / agreement assert)
    std::vector<double> pool_numeric_float_values; // [E] exact float/double payload
    std::vector<int64_t> pool_numeric_int_values;  // [E] exact integer payload
    std::vector<uint8_t> pool_numeric_kinds;       // [E] Tokenizer::NumericPayloadKind
    std::vector<int> pool_atom_types;          // [E] Tokenizer::AtomType enum values

    // Arg/option selector supervision (execution-independent). For each position t
    // whose NEXT token is an atom in the same row, the target is the batch-global
    // pool index of that next atom's entry (row_atom_offset[row] + atom_entry_ids[t+1])
    // — i.e. "which option should be selected at t". -1 = unsupervised (next token
    // is not a selectable atom). This is supervision only; token t+1's metadata is
    // never an input at t. Populated alongside the pool when the selector is enabled.
    std::vector<int> arg_select_targets;       // [total_tokens] batch-global pool index or -1
    int arg_select_valid_count = 0;            // number of supervised (>= 0) targets

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
        if (!isInferenceDecode()) {
            return true;
        }

        // Cached decode payloads normally carry the pending token and all of
        // its host-authored side channels. The legacy geometry-only decode
        // helper intentionally leaves every input array empty, so detect
        // ownership from the payload contents instead of rejecting all decode
        // payloads solely because of their mode.
        return !input_ids.empty() ||
               !numeric_values.empty() ||
               !atom_mask.empty() ||
               !atom_aux_target_mask.empty() ||
               !atom_flags.empty() ||
               !atom_entry_ids.empty() ||
               !token_to_slot_index_map.empty();
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

    GRIM::GoalSpanView goalSpansForRow(std::size_t row) const;
    const uint8_t* atomAuxTargetMaskForRow(std::size_t row) const;

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
                " (must be > 0)");
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
        if (isTraining() &&
            (static_cast<int>(prompt_lengths.size()) != batch_size ||
             static_cast<int>(prompt_end_positions.size()) != batch_size)) {
            throw std::runtime_error(
                std::string(caller) +
                ": training prompt-boundary arrays must both have batch_size entries");
        }
        if (isTraining() && static_cast<int>(goals.size()) != batch_size) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.goals.size()=" +
                std::to_string(goals.size()) + " != batch_size=" +
                std::to_string(batch_size));
        }
        if (isInference() && !goals.empty()) {
            throw std::runtime_error(
                std::string(caller) +
                ": inference BatchPayload must not carry training-row goal metadata");
        }
        if (!prompt_lengths.empty() || !prompt_end_positions.empty()) {
            if (static_cast<int>(prompt_lengths.size()) != batch_size ||
                static_cast<int>(prompt_end_positions.size()) != batch_size) {
                throw std::runtime_error(
                    std::string(caller) + ": prompt-boundary array size mismatch");
            }
            for (int b = 0; b < batch_size; ++b) {
                const int32_t length = prompt_lengths[static_cast<std::size_t>(b)];
                const int32_t end = prompt_end_positions[static_cast<std::size_t>(b)];
                if (length == 0) {
                    if (end != -1) {
                        throw std::runtime_error(
                            std::string(caller) +
                            ": empty prompt span requires end=-1 at batch row " +
                            std::to_string(b));
                    }
                    continue;
                }
                if (length < 0 || end < 0 ||
                    end >= seq_lengths[static_cast<std::size_t>(b)] ||
                    end - length + 1 < 0) {
                    throw std::runtime_error(
                        std::string(caller) + ": invalid prompt span at batch row " +
                        std::to_string(b));
                }
            }
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
        if (isTraining() && static_cast<int>(atom_aux_target_mask.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_aux_target_mask.size()=" +
                std::to_string(atom_aux_target_mask.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (isInference() && !atom_aux_target_mask.empty()) {
            throw std::runtime_error(
                std::string(caller) +
                ": inference BatchPayload must not carry training-only atom auxiliary ownership");
        }
        if (ownsHostInputData() && static_cast<int>(atom_flags.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_flags.size()=" +
                std::to_string(atom_flags.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (ownsHostInputData() && static_cast<int>(atom_entry_ids.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_entry_ids.size()=" +
                std::to_string(atom_entry_ids.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (ownsHostInputData()) {
            for (int i = 0; i < total_tokens; ++i) {
                if (atom_mask[i] > 1) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload.atom_mask[" +
                        std::to_string(i) + "]=" + std::to_string(atom_mask[i]) +
                        " (must be binary)");
                }
                const bool has_atom_metadata =
                    atom_mask[i] != 0 ||
                    atom_entry_ids[i] != GRIM::Tokenizer::kAtomEntryNone ||
                    numeric_values[i] != 0.0f ||
                    atom_flags[i] != 0;
                if (has_atom_metadata &&
                    !GRIM::Tokenizer::isAtomOpenTokenId(input_ids[i])) {
                    throw std::runtime_error(
                        std::string(caller) +
                        ": atom metadata is present outside a typed opening boundary at index=" +
                        std::to_string(i) + " token_id=" +
                        std::to_string(input_ids[i]));
                }
            }
        }
        if (ownsHostInputData() && static_cast<int>(token_to_slot_index_map.size()) != total_tokens) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.token_to_slot_index_map.size()=" +
                std::to_string(token_to_slot_index_map.size()) + " != total_tokens=" +
                std::to_string(total_tokens));
        }
        if (static_cast<int>(atom_positions.size()) != static_cast<int>(atom_types.size())) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.atom_positions.size()=" +
                std::to_string(atom_positions.size()) + " != atom_types.size()=" +
                std::to_string(atom_types.size()));
        }
        if (ownsHostInputData()) {
            std::size_t compact_atom_index = 0;
            for (int row = 0; row < batch_size; ++row) {
                bool inside_atom = false;
                GRIM::Tokenizer::AtomType open_type =
                    GRIM::Tokenizer::AtomType::ATOM_INT;
                const int row_offset = row * max_seq_len;
                const int row_length = seq_lengths[static_cast<std::size_t>(row)];
                for (int token = 0; token < max_seq_len; ++token) {
                    const int flat_position = row_offset + token;
                    if (token >= row_length) {
                        if (isTraining() &&
                            atom_aux_target_mask[static_cast<std::size_t>(flat_position)] != 0) {
                            throw std::runtime_error(
                                std::string(caller) +
                                ": atom auxiliary ownership is present on padding at flat position=" +
                                std::to_string(flat_position));
                        }
                        continue;
                    }

                    const int token_id = input_ids[static_cast<std::size_t>(flat_position)];
                    uint8_t expected_aux_owner = inside_atom ? 1 : 0;
                    if (GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
                        if (inside_atom) {
                            throw std::runtime_error(
                                std::string(caller) +
                                ": nested typed atom opening at flat position=" +
                                std::to_string(flat_position));
                        }
                        if (atom_mask[static_cast<std::size_t>(flat_position)] != 1) {
                            throw std::runtime_error(
                                std::string(caller) +
                                ": typed atom opening is missing atom_mask=1 at flat position=" +
                                std::to_string(flat_position));
                        }
                        inside_atom = true;
                        open_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
                        expected_aux_owner = 1;

                        if (compact_atom_index >= atom_positions.size() ||
                            atom_positions[compact_atom_index] != flat_position ||
                            atom_types[compact_atom_index] != static_cast<int>(open_type)) {
                            throw std::runtime_error(
                                std::string(caller) +
                                ": compact atom opening metadata does not match flat position=" +
                                std::to_string(flat_position));
                        }
                        ++compact_atom_index;
                    } else if (GRIM::Tokenizer::isAtomCloseTokenId(token_id)) {
                        if (!inside_atom ||
                            GRIM::Tokenizer::tokenIdToAtomType(token_id) != open_type) {
                            throw std::runtime_error(
                                std::string(caller) +
                                ": unmatched or mismatched typed atom close at flat position=" +
                                std::to_string(flat_position));
                        }
                        inside_atom = false;
                        expected_aux_owner = 0;
                    }

                    if (isTraining() &&
                        atom_aux_target_mask[static_cast<std::size_t>(flat_position)] !=
                            expected_aux_owner) {
                        throw std::runtime_error(
                            std::string(caller) +
                            ": atom_aux_target_mask disagrees with typed span at flat position=" +
                            std::to_string(flat_position) + " expected=" +
                            std::to_string(expected_aux_owner) + " actual=" +
                            std::to_string(atom_aux_target_mask[
                                static_cast<std::size_t>(flat_position)]));
                    }
                }
                if (inside_atom) {
                    throw std::runtime_error(
                        std::string(caller) +
                        ": typed atom opening has no close in batch row=" +
                        std::to_string(row));
                }
            }
            if (compact_atom_index != atom_positions.size()) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": compact atom metadata contains entries not backed by typed openings");
            }
        }
        // Number auxiliary target geometry (compact, atom-aligned).
        if (number_aux_target_digit_slots < 0 ||
            number_aux_target_max_abs_pow10 < 0 ||
            number_aux_target_valid_count < 0) {
            throw std::runtime_error(
                std::string(caller) +
                ": BatchPayload number auxiliary target geometry contains a negative value");
        }
        const std::size_t target_atoms = atom_positions.size();
        const std::size_t target_slots =
            static_cast<std::size_t>(number_aux_target_digit_slots);
        auto requireNumberTargetSize =
            [&](std::size_t actual, std::size_t expected, const char* name) {
                if (actual != expected) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload." + name + ".size()=" +
                        std::to_string(actual) + " != expected=" +
                        std::to_string(expected));
                }
            };
        if (number_aux_target_digit_slots == 0) {
            if (number_aux_target_max_abs_pow10 != 0 ||
                number_aux_target_valid_count != 0 ||
                !number_aux_target_valid.empty() ||
                !number_aux_target_sign_negative.empty() ||
                !number_aux_target_base.empty() ||
                !number_aux_target_digit_count.empty() ||
                !number_aux_target_digits.empty() ||
                !number_aux_target_pow10_index.empty() ||
                !number_aux_target_digit_mask.empty() ||
                !number_aux_target_float_values.empty() ||
                !number_aux_target_int_values.empty() ||
                !number_aux_target_numeric_kinds.empty()) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": number auxiliary targets are populated while digit_slots=0");
            }
        } else {
            requireNumberTargetSize(number_aux_target_valid.size(), target_atoms,
                                    "number_aux_target_valid");
            requireNumberTargetSize(number_aux_target_sign_negative.size(), target_atoms,
                                    "number_aux_target_sign_negative");
            requireNumberTargetSize(number_aux_target_base.size(), target_atoms,
                                    "number_aux_target_base");
            requireNumberTargetSize(number_aux_target_digit_count.size(), target_atoms,
                                    "number_aux_target_digit_count");
            requireNumberTargetSize(number_aux_target_float_values.size(), target_atoms,
                                    "number_aux_target_float_values");
            requireNumberTargetSize(number_aux_target_int_values.size(), target_atoms,
                                    "number_aux_target_int_values");
            requireNumberTargetSize(number_aux_target_numeric_kinds.size(), target_atoms,
                                    "number_aux_target_numeric_kinds");
            requireNumberTargetSize(number_aux_target_digits.size(), target_atoms * target_slots,
                                    "number_aux_target_digits");
            requireNumberTargetSize(number_aux_target_pow10_index.size(), target_atoms * target_slots,
                                    "number_aux_target_pow10_index");
            requireNumberTargetSize(number_aux_target_digit_mask.size(), target_atoms * target_slots,
                                    "number_aux_target_digit_mask");

            int counted_valid = 0;
            for (std::size_t atom = 0; atom < target_atoms; ++atom) {
                const bool numeric = GRIM::Tokenizer::isNumericAtom(
                    static_cast<GRIM::Tokenizer::AtomType>(atom_types[atom]));
                const uint8_t valid = number_aux_target_valid[atom];
                if (valid > 1) {
                    throw std::runtime_error(
                        std::string(caller) +
                        ": number_aux_target_valid must be binary at atom=" +
                        std::to_string(atom));
                }
                counted_valid += valid != 0 ? 1 : 0;
                const std::size_t digit_count = number_aux_target_digit_count[atom];
                if (!numeric) {
                    if (valid != 0 || digit_count != 0 ||
                        number_aux_target_base[atom] != 0) {
                        throw std::runtime_error(
                            std::string(caller) +
                            ": non-numeric atom carries a number auxiliary target at atom=" +
                            std::to_string(atom));
                    }
                    continue;
                }
                if (digit_count == 0 || digit_count > target_slots ||
                    number_aux_target_base[atom] != 10) {
                    throw std::runtime_error(
                        std::string(caller) +
                        ": invalid numeric auxiliary target geometry at atom=" +
                        std::to_string(atom));
                }
                for (std::size_t slot = 0; slot < target_slots; ++slot) {
                    const std::size_t index = atom * target_slots + slot;
                    const uint8_t expected_mask = slot < digit_count ? 1 : 0;
                    if (number_aux_target_digit_mask[index] != expected_mask) {
                        throw std::runtime_error(
                            std::string(caller) +
                            ": number auxiliary digit mask disagrees with digit_count at atom=" +
                            std::to_string(atom) + " slot=" + std::to_string(slot));
                    }
                    if (expected_mask != 0 &&
                        (number_aux_target_digits[index] < 0 ||
                         number_aux_target_digits[index] > 9 ||
                         number_aux_target_pow10_index[index] < 0 ||
                         number_aux_target_pow10_index[index] >
                             2 * number_aux_target_max_abs_pow10)) {
                        throw std::runtime_error(
                            std::string(caller) +
                            ": number auxiliary digit or pow10 target is out of range at atom=" +
                            std::to_string(atom) + " slot=" + std::to_string(slot));
                    }
                }
            }
            if (counted_valid != number_aux_target_valid_count) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": number_aux_target_valid_count does not match target mask");
            }
        }
        // NumberEncoder digit-place channel geometry (compact, atom-aligned)
        if (number_encoder_digit_slots < 0) {
            throw std::runtime_error(
                std::string(caller) + ": BatchPayload.number_encoder_digit_slots=" +
                std::to_string(number_encoder_digit_slots) + " is negative");
        }
        if (number_encoder_digit_slots == 0) {
            if (!atom_digit_values.empty() || !atom_digit_pow10_index.empty() ||
                !atom_digit_mask.empty() || !atom_digit_slot_features.empty() ||
                !atom_global_features.empty()) {
                throw std::runtime_error(
                    std::string(caller) + ": BatchPayload number-encoder channels are populated "
                    "while number_encoder_digit_slots=0");
            }
        } else {
            const std::size_t atoms = atom_positions.size();
            const std::size_t slots = static_cast<std::size_t>(number_encoder_digit_slots);
            auto requireChannelSize = [&](std::size_t actual, std::size_t expected, const char* name) {
                if (actual != expected) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload." + name + ".size()=" +
                        std::to_string(actual) + " != expected=" + std::to_string(expected) +
                        " (atoms=" + std::to_string(atoms) +
                        ", digit_slots=" + std::to_string(slots) + ")");
                }
            };
            requireChannelSize(atom_digit_values.size(), atoms * slots, "atom_digit_values");
            requireChannelSize(atom_digit_pow10_index.size(), atoms * slots, "atom_digit_pow10_index");
            requireChannelSize(atom_digit_mask.size(), atoms * slots, "atom_digit_mask");
            requireChannelSize(atom_digit_slot_features.size(),
                               atoms * slots * static_cast<std::size_t>(kNumberSlotFeatureDim),
                               "atom_digit_slot_features");
            requireChannelSize(atom_global_features.size(),
                               atoms * static_cast<std::size_t>(kNumberGlobalFeatureDim),
                               "atom_global_features");
        }
        const std::size_t pool_entries = static_cast<std::size_t>(num_pool_atoms);
        auto requirePoolChannelSize =
            [&](std::size_t actual, std::size_t expected, const char* name) {
                if (actual != expected) {
                    throw std::runtime_error(
                        std::string(caller) + ": BatchPayload." + name + ".size()=" +
                        std::to_string(actual) + " != expected=" +
                        std::to_string(expected));
                }
            };
        requirePoolChannelSize(pool_numeric_values.size(), pool_entries, "pool_numeric_values");
        requirePoolChannelSize(
            pool_numeric_float_values.size(), pool_entries, "pool_numeric_float_values");
        requirePoolChannelSize(
            pool_numeric_int_values.size(), pool_entries, "pool_numeric_int_values");
        requirePoolChannelSize(pool_numeric_kinds.size(), pool_entries, "pool_numeric_kinds");
        requirePoolChannelSize(pool_atom_types.size(), pool_entries, "pool_atom_types");
        if (row_atom_offset.empty()) {
            if (num_pool_atoms != 0 || !arg_select_targets.empty() ||
                arg_select_valid_count != 0) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": selector pool metadata is partially populated");
            }
        } else {
            if (row_atom_offset.size() !=
                static_cast<std::size_t>(batch_size) + 1) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": BatchPayload.row_atom_offset.size() must equal batch_size + 1");
            }
            if (row_atom_offset.front() != 0 ||
                row_atom_offset.back() != num_pool_atoms) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": BatchPayload.row_atom_offset does not span num_pool_atoms");
            }
            for (std::size_t i = 1; i < row_atom_offset.size(); ++i) {
                if (row_atom_offset[i] < row_atom_offset[i - 1]) {
                    throw std::runtime_error(
                        std::string(caller) +
                        ": BatchPayload.row_atom_offset is not monotonic");
                }
            }
            requirePoolChannelSize(
                arg_select_targets.size(),
                static_cast<std::size_t>(total_tokens),
                "arg_select_targets");
            const int realized_valid_count = static_cast<int>(std::count_if(
                arg_select_targets.begin(),
                arg_select_targets.end(),
                [](int target) { return target >= 0; }));
            if (realized_valid_count != arg_select_valid_count) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": BatchPayload.arg_select_valid_count does not match targets");
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
    size_t atomPositionBytes() const { return atom_positions.size() * sizeof(int); }
    size_t atomTypeBytes()     const { return atom_types.size() * sizeof(int); }
    int authoredAtomCount() const { return static_cast<int>(atom_positions.size()); }
    size_t totalTransferBytes() const {
        return inputIdBytes() + targetIdBytes() + numericValueBytes() +
               atomMaskBytes() + atomFlagBytes() + slotMapBytes() +
               atomPositionBytes() + atomTypeBytes();
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
 * @param vocab_size     Model token-space width for target validation
 * @param batch_size         Fixed training batch size / row capacity
 * @param max_cached_seq_len GPU cache sequence length capacity
 * @param selector_enabled   Whether to materialize selector candidate metadata
 * @param number_encoder_digit_slots   Shared NumberEncoder/number-target digit capacity (0 = disabled)
 * @param number_encoder_max_abs_pow10 Shared NumberEncoder/number-target pow10 half-range
 * @return Complete BatchPayload ready for downstream consumption
 */
BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t batch_size,
    size_t max_cached_seq_len,
    bool selector_enabled,
    int number_encoder_digit_slots,
    int number_encoder_max_abs_pow10);

/**
 * Build a validated single-row inference prefill payload from tokenizer-authored
 * metadata. This is the inference data-ingestion boundary: callers provide all
 * per-token atom side channels explicitly, and downstream CUDA code consumes the
 * resulting BatchPayload + BatchDeviceBindings pair. Non-negative slot bindings
 * must lie in [execution_num_scratch_slots, execution_num_slots).
 */
BatchPayload buildInferenceBatchPayload(
    const std::vector<int>& token_ids,
    const std::vector<float>& numeric_values,
    const std::vector<uint8_t>& atom_mask,
    const std::vector<uint32_t>& atom_flags,
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table,
    const std::vector<uint32_t>& atom_entry_ids,
    const std::vector<int32_t>& token_to_slot_index_map,
    int vocab_size,
    size_t batch_capacity,
    size_t max_cached_seq_len,
    int execution_num_slots,
    int execution_num_scratch_slots,
    bool selector_enabled,
    int number_encoder_digit_slots,
    int number_encoder_max_abs_pow10);

/** Build a single-token inference decode geometry payload. */
BatchPayload buildInferenceDecodePayload(int vocab_size);

}  // namespace Batching
}  // namespace GRIM
