//======================================================//
//  BatchPayload.cu
//  Builder implementation for BatchPayload
//
//  Merges logic from:
//    - Phase2_TrainingLoop.cu::processBatch() sequence extraction (lines 2700-2730)
//    - deleted loss-local prepareLossBatchInputs() padding/masking
//    - legacy token-stat recomputation for gradient clipping
//
//  All metadata computed ONCE here. No downstream recomputation.
//  Rule 20: No fallbacks, crash on any inconsistency.
//======================================================//

#include "BatchPayload.hpp"
#include "Batching_GPU.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"
#include "../TokenizerArtifacts/GrmtSequence.hpp"
#include "../Execution/ExecutionPayloadValidation.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
namespace GRIM {
namespace Batching {

namespace {

void requirePositiveVocab(int vocab_size, const char* caller)
{
    if (vocab_size <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": vocab_size=" + std::to_string(vocab_size) +
            " (must be > 0)");
    }
}

BatchPayload makeInferenceBasePayload(
    int seq_len,
    int vocab_size,
    size_t batch_capacity,
    size_t max_cached_seq_len,
    BatchPayloadMode mode,
    bool row_execution_active,
    const char* caller)
{
    if (mode == BatchPayloadMode::Training) {
        throw std::runtime_error(std::string(caller) + ": inference builder received Training mode");
    }
    if (seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": seq_len=" + std::to_string(seq_len) +
                                 " (must be > 0)");
    }
    requirePositiveVocab(vocab_size, caller);
    if (batch_capacity == 0) {
        throw std::runtime_error(std::string(caller) + ": batch_capacity=0");
    }
    if (max_cached_seq_len == 0) {
        throw std::runtime_error(std::string(caller) + ": max_cached_seq_len=0");
    }
    if (static_cast<size_t>(seq_len) > max_cached_seq_len) {
        throw std::runtime_error(
            std::string(caller) + ": seq_len=" + std::to_string(seq_len) +
            " exceeds max_cached_seq_len=" + std::to_string(max_cached_seq_len));
    }
    if (batch_capacity < 1) {
        throw std::runtime_error(std::string(caller) + ": inference requires cache batch capacity >= 1");
    }

    BatchPayload payload;
    payload.mode = mode;
    payload.seq_ids.assign(1, 0);
    payload.batch_size = 1;
    payload.max_seq_len = seq_len;
    payload.total_tokens = seq_len;
    payload.actual_tokens = seq_len;
    payload.padding_tokens = 0;
    payload.valid_tokens = 0;
    payload.lm_valid_tokens = 0;
    payload.vocab_size = vocab_size;
    payload.seq_lengths.assign(1, seq_len);
    payload.valid_target_counts.assign(1, 0);
    payload.fits_in_cache = true;

    payload.execution_active.assign(1, row_execution_active);
    payload.compiled_bootstrap_bindings.resize(1);
    payload.teacher_steps.resize(1);
    payload.teacher_step_mask.resize(1);

    return payload;
}

void materializeAuthoredAtomFacts(
    BatchPayload& payload,
    const char* caller)
{
    payload.atom_positions.clear();
    payload.atom_types.clear();

    if (!payload.ownsHostInputData()) {
        return;
    }

    payload.atom_positions.reserve(static_cast<std::size_t>(payload.actual_tokens));
    payload.atom_types.reserve(static_cast<std::size_t>(payload.actual_tokens));

    for (int token_pos = 0; token_pos < payload.total_tokens; ++token_pos) {
        if (payload.atom_mask[static_cast<std::size_t>(token_pos)] == 0) {
            continue;
        }

        const int token_id = payload.input_ids[static_cast<std::size_t>(token_pos)];
        if (!GRIM::Tokenizer::isAtomTokenId(token_id)) {
            throw std::runtime_error(
                std::string(caller) + ": atom_mask marks token position " +
                std::to_string(token_pos) + " as atom but token_id=" +
                std::to_string(token_id) + " is not an atom placeholder");
        }
        const auto atom_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        if (payload.atom_entry_ids[static_cast<std::size_t>(token_pos)] == GRIM::Tokenizer::kAtomEntryNone) {
            throw std::runtime_error(
                std::string(caller) + ": atom_mask marks token position " +
                std::to_string(token_pos) + " as atom but atom_entry_ids is kAtomEntryNone");
        }

        payload.atom_positions.push_back(token_pos);
        payload.atom_types.push_back(static_cast<int>(atom_type));
    }
}

// =============================================================================
// materializeNumberEncoderChannels — digit-place input channels for the
// NumberEncoder numeric-meaning path (docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md).
//
// CAUSALITY: every channel row describes the CURRENT token's own arg_number
// metadata. Target-side (t+1) atom metadata is supervision-only and must never
// flow through this input boundary.
//
// FAIL-LOUD CONTRACT (Rule 20):
//   - numeric atom token without a resolvable AtomTable entry        -> throw
//   - numeric atom entry without arg_number metadata                 -> throw
//   - mantissa digit count exceeding the configured slot capacity    -> throw
//   - digit pow10 outside the configured ±max_abs_pow10 bucket range -> throw
// Padding digit slots are mask=0 and contribute exactly zero downstream.
// =============================================================================
void materializeNumberEncoderChannels(
    BatchPayload& payload,
    int digit_slots,
    int max_abs_pow10,
    const char* caller)
{
    payload.number_encoder_digit_slots = 0;
    payload.atom_digit_values.clear();
    payload.atom_digit_pow10_index.clear();
    payload.atom_digit_mask.clear();
    payload.atom_digit_slot_features.clear();
    payload.atom_global_features.clear();

    if (digit_slots <= 0) {
        return;  // NumberEncoder disabled by config — channels stay empty.
    }
    if (max_abs_pow10 <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": number_encoder digit_slots=" + std::to_string(digit_slots) +
            " but max_abs_pow10=" + std::to_string(max_abs_pow10) + " is not positive");
    }
    if (!payload.ownsHostInputData()) {
        throw std::runtime_error(
            std::string(caller) + ": number-encoder channels requested for a payload mode (" +
            payload.modeName() + ") that owns no host input data");
    }

    const std::size_t atoms = payload.atom_positions.size();
    const std::size_t slots = static_cast<std::size_t>(digit_slots);
    payload.number_encoder_digit_slots = digit_slots;
    payload.atom_digit_values.assign(atoms * slots, 0);
    payload.atom_digit_pow10_index.assign(atoms * slots, 0);
    payload.atom_digit_mask.assign(atoms * slots, 0.0f);
    payload.atom_digit_slot_features.assign(
        atoms * slots * static_cast<std::size_t>(BatchPayload::kNumberSlotFeatureDim), 0.0f);
    payload.atom_global_features.assign(
        atoms * static_cast<std::size_t>(BatchPayload::kNumberGlobalFeatureDim), 0.0f);

    const float pow10_norm_scale = 1.0f / static_cast<float>(max_abs_pow10);
    const float digit_count_norm_scale = 1.0f / static_cast<float>(digit_slots);

    for (std::size_t atom_idx = 0; atom_idx < atoms; ++atom_idx) {
        const int token_pos = payload.atom_positions[atom_idx];
        const int row = token_pos / payload.max_seq_len;
        if (row < 0 || row >= static_cast<int>(payload.seq_atom_tables.size())) {
            throw std::runtime_error(
                std::string(caller) + ": number-encoder atom_idx=" + std::to_string(atom_idx) +
                " token_pos=" + std::to_string(token_pos) + " resolves to batch row " +
                std::to_string(row) + " outside seq_atom_tables.size()=" +
                std::to_string(payload.seq_atom_tables.size()));
        }
        const auto& atom_table = payload.seq_atom_tables[static_cast<std::size_t>(row)];
        if (!atom_table) {
            throw std::runtime_error(
                std::string(caller) + ": number-encoder batch row " + std::to_string(row) +
                " has a NULL AtomTable while atom token_pos=" + std::to_string(token_pos) +
                " requires arg_number metadata");
        }
        const uint32_t entry_id = payload.atom_entry_ids[static_cast<std::size_t>(token_pos)];
        if (entry_id == GRIM::Tokenizer::kAtomEntryNone) {
            throw std::runtime_error(
                std::string(caller) + ": number-encoder atom token_pos=" + std::to_string(token_pos) +
                " carries kAtomEntryNone — upstream tokenization must register numeric atoms");
        }
        const auto entry = atom_table->getAtom(entry_id);
        if (!entry.has_value()) {
            throw std::runtime_error(
                std::string(caller) + ": number-encoder atom_entry_id=" + std::to_string(entry_id) +
                " at token_pos=" + std::to_string(token_pos) + " is not retrievable from its AtomTable");
        }
        if (!GRIM::Tokenizer::isNumericAtom(entry->type)) {
            throw std::runtime_error(
                std::string(caller) + ": number-encoder atom_entry_id=" + std::to_string(entry_id) +
                " at token_pos=" + std::to_string(token_pos) + " is not a numeric atom type");
        }
        if (!entry->arg_number.has_value()) {
            throw std::runtime_error(
                std::string(caller) + ": numeric atom_entry_id=" + std::to_string(entry_id) +
                " at token_pos=" + std::to_string(token_pos) +
                " is missing required arg_number metadata");
        }
        const GRIM::Tokenizer::ArgNumber& number = *entry->arg_number;
        const std::size_t digit_count = number.digits.size();
        if (digit_count == 0) {
            throw std::runtime_error(
                std::string(caller) + ": numeric atom_entry_id=" + std::to_string(entry_id) +
                " has zero mantissa digit bindings");
        }
        if (digit_count > slots) {
            throw std::runtime_error(
                std::string(caller) + ": numeric atom_entry_id=" + std::to_string(entry_id) +
                " has " + std::to_string(digit_count) +
                " mantissa digits exceeding number_encoder_max_digit_slots=" +
                std::to_string(digit_slots) + " — refusing to silently truncate digit structure");
        }

        const float sign = number.sign_negative ? -1.0f : 1.0f;
        const bool is_float_atom = entry->type == GRIM::Tokenizer::AtomType::ATOM_FLOAT;
        const float is_float = is_float_atom ? 1.0f : 0.0f;

        const std::size_t slot_base = atom_idx * slots;
        for (std::size_t i = 0; i < digit_count; ++i) {
            const auto& binding = number.digits[i];
            if (binding.digit > 9) {
                throw std::runtime_error(
                    std::string(caller) + ": numeric atom_entry_id=" + std::to_string(entry_id) +
                    " digit[" + std::to_string(i) + "]=" + std::to_string(binding.digit) +
                    " is not a base-10 digit");
            }
            const int pow10 = static_cast<int>(binding.pow10);
            if (pow10 < -max_abs_pow10 || pow10 > max_abs_pow10) {
                throw std::runtime_error(
                    std::string(caller) + ": numeric atom_entry_id=" + std::to_string(entry_id) +
                    " digit[" + std::to_string(i) + "] pow10=" + std::to_string(pow10) +
                    " outside configured number_encoder_max_abs_pow10=±" +
                    std::to_string(max_abs_pow10));
            }
            const std::size_t slot = slot_base + i;
            payload.atom_digit_values[slot] = static_cast<int>(binding.digit);
            payload.atom_digit_pow10_index[slot] = pow10 + max_abs_pow10;  // bucket [0, 2*max]
            payload.atom_digit_mask[slot] = 1.0f;

            float* feat = payload.atom_digit_slot_features.data() +
                          slot * static_cast<std::size_t>(BatchPayload::kNumberSlotFeatureDim);
            feat[0] = static_cast<float>(binding.digit) / 9.0f;
            feat[1] = static_cast<float>(pow10) * pow10_norm_scale;
            feat[2] = binding.digit == 0 ? 1.0f : 0.0f;
            feat[3] = sign;
            feat[4] = is_float;
        }

        float* gfeat = payload.atom_global_features.data() +
                       atom_idx * static_cast<std::size_t>(BatchPayload::kNumberGlobalFeatureDim);
        gfeat[0] = sign;
        gfeat[1] = static_cast<float>(number.exponent_value) * pow10_norm_scale;
        gfeat[2] = static_cast<float>(number.integer_digit_count) * digit_count_norm_scale;
        gfeat[3] = static_cast<float>(number.fractional_digit_count) * digit_count_norm_scale;
        gfeat[4] = number.has_decimal_point ? 1.0f : 0.0f;
        gfeat[5] = is_float;
    }
}

}  // namespace

BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t batch_size,
    size_t max_cached_seq_len,
    int execution_num_slots,
    int execution_num_ops,
    int execution_num_steps,
    int mtp_k,
    int number_encoder_digit_slots,
    int number_encoder_max_abs_pow10)
{
    BatchPayload payload;
    payload.mode = BatchPayloadMode::Training;

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 1: Identity — carry forward from assignment
    // ═════════════════════════════════════════════════════════════════════════
    payload.seq_ids = assignment.seq_ids;
    payload.batch_size = static_cast<int>(assignment.seq_ids.size());

    if (payload.batch_size <= 0) {
        throw std::runtime_error(
            "buildBatchPayload: batch has 0 sequences — scheduler produced empty batch");
    }
    if (static_cast<size_t>(payload.batch_size) != batch_size) {
        throw std::runtime_error(
            "buildBatchPayload: assignment batch_size=" + std::to_string(payload.batch_size) +
            " != fixed batch_size=" + std::to_string(batch_size) +
            " — training uses fixed batch rows; scheduler must emit full batches");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 2: Extract sequences and compute geometry in one pass
    // ═════════════════════════════════════════════════════════════════════════

    // Temporary ragged storage — we extract once, then flatten into padded layout
    struct RawSeq {
        const std::vector<int>* token_ids;
        const std::vector<int>* targets;
        const std::vector<float>* numeric_values;
        const std::vector<uint8_t>* atom_mask;
        const std::vector<uint32_t>* atom_flags;
        std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table;
        const std::vector<uint32_t>* atom_entry_ids;
        const std::vector<int32_t>* exec_slots;
        int length;

        // Compiled execution metadata (per-row, not per-token)
        bool execution_active;
        const std::vector<GRIM::Execution::CompiledBootstrapBinding>* compiled_bootstrap_bindings;
        const std::vector<GRIM::Execution::TeacherStep>* teacher_steps;
    };

    std::vector<RawSeq> raw;
    raw.reserve(payload.batch_size);

    payload.seq_lengths.resize(payload.batch_size);
    payload.max_seq_len = 0;
    payload.actual_tokens = 0;

    for (int b = 0; b < payload.batch_size; ++b) {
        const uint32_t sid = payload.seq_ids[b];

        // Rule 20: crash if view is out of bounds or null
        if (sid >= views.size()) {
            throw std::runtime_error(
                "buildBatchPayload: seq_id=" + std::to_string(sid) +
                " exceeds views.size()=" + std::to_string(views.size()));
        }
        const GRIM::TokenizerArtifacts::GrmtSequence* seq = views[sid];
        if (!seq) {
            throw std::runtime_error(
                "buildBatchPayload: views[" + std::to_string(sid) + "] is NULL");
        }

        const int seq_len = static_cast<int>(seq->token_ids.size());
        if (seq_len <= 0) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) + " has 0 tokens");
        }

        // Validate alignment between all per-token arrays
        if (static_cast<int>(seq->targets.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " token_ids.size()=" + std::to_string(seq_len) +
                " != targets.size()=" + std::to_string(seq->targets.size()));
        }
        if (static_cast<int>(seq->token_numeric_values.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " numeric_values.size()=" + std::to_string(seq->token_numeric_values.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->token_atom_mask.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " atom_mask.size()=" + std::to_string(seq->token_atom_mask.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->atom_entry_ids.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " atom_entry_ids.size()=" + std::to_string(seq->atom_entry_ids.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->token_atom_flags.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " token_atom_flags.size()=" + std::to_string(seq->token_atom_flags.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }

        // Validate per-token IDs against the loss contract:
        //   input_id MUST be in [0, vocab_size)
        //   target  MUST be in {-1} ∪ [0, vocab_size)
        // Catching this here prevents OOB GPU reads in CE / embedding lookup.
        for (int t = 0; t < seq_len; ++t) {
            const int iid = seq->token_ids[t];
            if (iid < 0 || iid >= vocab_size) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " has invalid input token " + std::to_string(iid) +
                    " at position " + std::to_string(t) +
                    " (must be in [0, " + std::to_string(vocab_size) + "))");
            }
            const int tid = seq->targets[t];
            // Loss contract: -1 (masked) is the ONLY legal negative value.
            if (tid < -1 || tid >= vocab_size) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " has invalid target token " + std::to_string(tid) +
                    " at position " + std::to_string(t) +
                    " (must be -1 or in [0, " + std::to_string(vocab_size) + "))");
            }

            const bool token_is_atom = token_layout.isAtom(iid);
            const bool mask_is_atom = seq->token_atom_mask[t] != 0;
            const bool entry_is_atom = seq->atom_entry_ids[t] != GRIM::Tokenizer::kAtomEntryNone;
            const bool flags_are_atom = seq->token_atom_flags[t] != 0;
            if (token_is_atom != mask_is_atom) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " token_atom_mask mismatch at position " + std::to_string(t) +
                    " for input token " + std::to_string(iid) +
                    " (token_is_atom=" + std::to_string(token_is_atom) +
                    ", atom_mask=" + std::to_string(static_cast<int>(seq->token_atom_mask[t])) + ")");
            }
            if (token_is_atom != entry_is_atom) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " atom_entry_ids mismatch at position " + std::to_string(t) +
                    " for input token " + std::to_string(iid) +
                    " (token_is_atom=" + std::to_string(token_is_atom) +
                    ", atom_entry_id=" + std::to_string(seq->atom_entry_ids[t]) + ")");
            }
            if (!token_is_atom && flags_are_atom) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " has nonzero token_atom_flags=" + std::to_string(seq->token_atom_flags[t]) +
                    " at non-atom input token " + std::to_string(iid) +
                    " position " + std::to_string(t));
            }
        }

        const std::vector<int32_t>* exec_slots_ptr = nullptr;
        if (!seq->token_exec_slots.empty()) {
            if (static_cast<int>(seq->token_exec_slots.size()) != seq_len) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " token_exec_slots.size()=" + std::to_string(seq->token_exec_slots.size()) +
                    " != token_ids.size()=" + std::to_string(seq_len));
            }
            exec_slots_ptr = &seq->token_exec_slots;
        }

        raw.push_back({
            &seq->token_ids,
            &seq->targets,
            &seq->token_numeric_values,
            &seq->token_atom_mask,
            &seq->token_atom_flags,
            seq->atom_table,
            &seq->atom_entry_ids,
            exec_slots_ptr,
            seq_len,
            seq->execution_active,
            &seq->compiled_bootstrap_bindings,
            &seq->teacher_steps
        });

        payload.seq_lengths[b] = seq_len;
        payload.actual_tokens += seq_len;

    }

    // Pad every batch to the configured sliding-window / cache cap so the
    // training-time token rectangle is constant across batches. The sliding
    // window stage in Phase1 already guarantees seq_len <= max_cached_seq_len
    // for every row, so this is a pure pad-up (never a truncation).
    // Rule 20: total_tokens MUST be deterministic per run — do NOT collapse to
    // the per-batch max, which silently shrinks the rectangle for batches that
    // happen to contain only short sequences and breaks downstream consumers
    // (loss accounting, GPU cache fit, telemetry totals).
    payload.max_seq_len = static_cast<int>(max_cached_seq_len);

    // Defense-in-depth: any individual row exceeding the cap is a sliding-window
    // contract violation — fail loud rather than silently truncating.
    for (int b = 0; b < payload.batch_size; ++b) {
        if (payload.seq_lengths[b] > payload.max_seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: seq " + std::to_string(payload.seq_ids[b]) +
                " length=" + std::to_string(payload.seq_lengths[b]) +
                " exceeds max_cached_seq_len=" + std::to_string(payload.max_seq_len) +
                " — sliding window invariant violated");
        }
    }

    // Derived geometry
    payload.total_tokens = payload.batch_size * payload.max_seq_len;
    payload.padding_tokens = payload.total_tokens - payload.actual_tokens;
    payload.vocab_size = vocab_size;

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 3: Cache fit check
    // ═════════════════════════════════════════════════════════════════════════
    payload.fits_in_cache =
        (static_cast<size_t>(payload.batch_size) <= batch_size) &&
        (static_cast<size_t>(payload.max_seq_len) <= max_cached_seq_len);

    if (!payload.fits_in_cache) {
        const char* reason = (static_cast<size_t>(payload.batch_size) > batch_size)
            ? "BATCH_SIZE" : "SEQ_LEN";
        fprintf(stderr,
            "[buildBatchPayload] FATAL: batch does not fit cache (%s): "
            "batch=%d [limit=%zu], max_seq_len=%d [limit=%zu]\n",
            reason,
            payload.batch_size, batch_size,
            payload.max_seq_len, max_cached_seq_len);
        throw std::runtime_error(
            "buildBatchPayload: batch does not fit GPU cache (" +
            std::string(reason) + "): batch=" + std::to_string(payload.batch_size) +
            " limit=" + std::to_string(batch_size) +
            ", seq_len=" + std::to_string(payload.max_seq_len) +
            " limit=" + std::to_string(max_cached_seq_len));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 4: Pad all arrays to flat [batch_size * max_seq_len] layout
    //          and count valid targets — SINGLE PASS
    // ═════════════════════════════════════════════════════════════════════════
    const int S = payload.max_seq_len;
    const size_t flat_size = static_cast<size_t>(payload.total_tokens);

    payload.input_ids.assign(flat_size, Tokenizer::PAD_TOKEN_ID);  // PAD=1, NOT UNK=0
    payload.target_ids.assign(flat_size, -1);  // padding targets = masked
    payload.numeric_values.assign(flat_size, 0.0f);
    payload.atom_mask.assign(flat_size, 0);
    payload.atom_flags.assign(flat_size, 0);
    payload.atom_entry_ids.assign(flat_size, GRIM::Tokenizer::kAtomEntryNone);
    payload.token_to_slot_map.assign(flat_size, -1);
    payload.seq_atom_tables.resize(payload.batch_size);
    payload.valid_target_counts.resize(payload.batch_size, 0);

    // Compiled execution metadata arrays — sized to batch_size
    payload.execution_active.resize(payload.batch_size, false);
    payload.compiled_bootstrap_bindings.resize(payload.batch_size);
    payload.teacher_steps.resize(payload.batch_size);
    payload.teacher_step_mask.resize(payload.batch_size);

    payload.valid_tokens = 0;

    for (int b = 0; b < payload.batch_size; ++b) {
        const auto& r = raw[b];
        const int seq_len = r.length;
        const size_t row_offset = static_cast<size_t>(b) * S;

        // Bulk copy token IDs (contiguous source → contiguous destination row)
        std::memcpy(&payload.input_ids[row_offset],
                    r.token_ids->data(),
                    seq_len * sizeof(int));

        // Copy targets and enforce final-position autoregressive boundary.
        // Rule 20: upstream MUST already mark the final position with -1 (no
        // valid next-token target exists at the sequence boundary). We assert
        // that contract instead of silently overwriting — silent overwrite
        // would mask a real upstream bug where supervision was wrongly
        // assigned to the boundary token.
        if (r.targets->at(seq_len - 1) != -1) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(payload.seq_ids[b]) +
                " has non-(-1) target at final position " + std::to_string(seq_len - 1) +
                " (got target=" + std::to_string(r.targets->at(seq_len - 1)) +
                "). Upstream must mask the final-position target with -1; "
                "BatchPayload refuses to silently drop supervision.");
        }
        std::memcpy(&payload.target_ids[row_offset],
                    r.targets->data(),
                    seq_len * sizeof(int));
        // Padding positions beyond seq_len already have target=-1 from assign()
        // Defense-mask layout-only targets and count valid targets — SINGLE PASS.
        // UNK, PAD, and BOS are never valid prediction targets.
        // EOS IS a valid target — model must learn to predict end-of-sequence.
        // This is a safety net; DataLoader should already mask these.
        int valid_count = 0;
        for (int t = 0; t < seq_len - 1; ++t) {
            const int target = payload.target_ids[row_offset + t];
            if (target >= 0 && GRIM::Tokenizer::isNeverTargetSpecialTokenId(target)) {
                // Non-content token leaked through DataLoader — mask it
                payload.target_ids[row_offset + t] = -1;
            } else if (target >= 0) {
                ++valid_count;
            }
        }

        payload.valid_target_counts[b] = valid_count;
        payload.valid_tokens += valid_count;

        // Bulk copy numeric values
        std::memcpy(&payload.numeric_values[row_offset],
                    r.numeric_values->data(),
                    seq_len * sizeof(float));

        // Bulk copy atom mask
        std::memcpy(&payload.atom_mask[row_offset],
                    r.atom_mask->data(),
                    seq_len * sizeof(uint8_t));

        // Bulk copy atom flags (type-specific metadata from AtomTable)
        std::memcpy(&payload.atom_flags[row_offset],
                    r.atom_flags->data(),
                    seq_len * sizeof(uint32_t));

        // Copy atom entry IDs (bulk memcpy — fixed-size uint32_t)
        std::memcpy(&payload.atom_entry_ids[row_offset],
                    r.atom_entry_ids->data(),
                    seq_len * sizeof(uint32_t));

        // Copy execution slot map (runtime substrate metadata; -1 for non-state-bearing)
        if (r.exec_slots) {
            std::memcpy(&payload.token_to_slot_map[row_offset],
                        r.exec_slots->data(),
                        seq_len * sizeof(int32_t));
        }
        // else: stays -1 (default) — dataset doesn't yet provide slot assignments

        // Store AtomTable reference for this batch row
        payload.seq_atom_tables[b] = r.atom_table;

        // Compiled execution metadata (per-row)
        payload.execution_active[b] = r.execution_active;
        if (r.compiled_bootstrap_bindings && !r.compiled_bootstrap_bindings->empty()) {
            payload.compiled_bootstrap_bindings[b] = *r.compiled_bootstrap_bindings;
        }
        if (r.teacher_steps && !r.teacher_steps->empty()) {
            const int real_count = static_cast<int>(r.teacher_steps->size());
            payload.teacher_steps[b] = *r.teacher_steps;

            // Pad to execution_num_steps by repeating last step
            if (real_count < execution_num_steps) {
                const auto& last = payload.teacher_steps[b].back();
                payload.teacher_steps[b].resize(execution_num_steps, last);
            }

            // Build step mask: 1 = real step, 0 = padded step
            payload.teacher_step_mask[b].assign(execution_num_steps, 0);
            for (int k = 0; k < std::min(real_count, execution_num_steps); ++k) {
                payload.teacher_step_mask[b][k] = 1;
            }
        }
    }

    materializeAuthoredAtomFacts(payload, "buildBatchPayload");
    materializeNumberEncoderChannels(
        payload, number_encoder_digit_slots, number_encoder_max_abs_pow10, "buildBatchPayload");

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 4b: Execution-slot target masking
    //
    // Tokens owned by the execution block (token_to_slot_map[p] >= 0) are
    // supervised by the numeric head, NOT by LM cross-entropy.  Because the
    // DataLoader shift convention is target_ids[t] = token_ids[t+1] (the LM
    // at position t predicts the token at position t+1), the LM CE that
    // would supervise predicting an execution-owned token at position p lives
    // at target_ids[p-1].  We mask THAT position, not target_ids[p] (which
    // predicts the *next* token after the slot and is generally a normal LM
    // target).  If p is the first token in its row (p == row_start) there is
    // no in-row LM position predicting it, so nothing to mask.
    // Only execution-active rows can have valid slots.  Tokens that are atoms
    // but have NO slot (slot == -1) remain under LM CE — they are ordinary
    // numeric text, not execution-owned.
    // Accounting: each masked LM target decrements both valid_target_counts[b]
    // and the batch-level valid_tokens, so the validate() invariant
    // sum(valid_target_counts) == valid_tokens is preserved.  lm_valid_tokens
    // is then equal to the post-mask valid_tokens (kept as a distinct field so
    // downstream callers that read it continue to work unchanged).
    // ═════════════════════════════════════════════════════════════════════════
    {
        int slots_masked = 0;
        for (int b = 0; b < payload.batch_size; ++b) {
            if (!payload.execution_active[b])
                continue;  // Row has no execution supervision — no slots possible
            const int row_start = b * S;
            const int row_end   = row_start + payload.seq_lengths[b];
            for (int p = row_start; p < row_end; ++p) {
                if (payload.token_to_slot_map[p] < 0) continue;
                // The LM CE predicting position p lives at target_ids[p-1].
                // Skip when p is the row's first token (no in-row predictor).
                if (p == row_start) continue;
                const int pred_idx = p - 1;
                if (payload.target_ids[pred_idx] != -1) {
                    payload.target_ids[pred_idx] = -1;
                    payload.valid_target_counts[b]--;
                    slots_masked++;
                }
            }
        }
        payload.valid_tokens   -= slots_masked;
        payload.lm_valid_tokens = payload.valid_tokens;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 4c: MTP shifted targets (multi-token prediction)
    // For each MTP head k (shift = k+1), build shifted target array from the
    // ALREADY-MASKED target_ids.  This inherits execution-slot masking, final-
    // position masking, and non-content defense masking — all done above.
    // Shift rule per sequence row b:
    //   shifted[b*S + t] = target_ids[b*S + t + shift]  if (t + shift) < S
    //                     = -1                            otherwise (out of bounds)
    //
    // mtp_valid_counts[k] = count of shifted[t] != -1 across all tokens.
    // This is the AUTHORITATIVE valid count for MTP head k loss normalization.
    // ═════════════════════════════════════════════════════════════════════════
    if (mtp_k > 0) {
        payload.mtp_shifted_targets.resize(mtp_k);
        payload.mtp_valid_counts.resize(mtp_k, 0);

        for (int k = 0; k < mtp_k; ++k) {
            const int shift = k + 1;
            auto& shifted = payload.mtp_shifted_targets[k];
            shifted.assign(flat_size, -1);
            int valid_count = 0;

            for (int b = 0; b < payload.batch_size; ++b) {
                const int row_offset = b * S;
                const int seq_len = payload.seq_lengths[b];

                // Only positions where (t + shift) is within the sequence boundary
                // can have valid shifted targets.  Padding beyond seq_len already
                // has target_ids = -1, so shifted inherits that correctly.
                const int max_t = seq_len - shift;  // last valid source position
                for (int t = 0; t < max_t; ++t) {
                    const int src = payload.target_ids[row_offset + t + shift];
                    shifted[row_offset + t] = src;
                    if (src >= 0) ++valid_count;
                }
                // Positions [max_t, S) stay -1 from assign() above
            }

            payload.mtp_valid_counts[k] = valid_count;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 5: Final validation
    // ═════════════════════════════════════════════════════════════════════════
    if (payload.valid_tokens <= 0) {
        std::ostringstream msg;
        msg << "buildBatchPayload: valid_tokens=0 after padding. "
            << "batch_size=" << payload.batch_size
            << " max_seq_len=" << payload.max_seq_len
            << " total_tokens=" << payload.total_tokens
            << " seq_ids=[";
        for (size_t i = 0; i < payload.seq_ids.size(); ++i) {
            msg << payload.seq_ids[i];
            if (i + 1 < payload.seq_ids.size()) msg << ",";
        }
        msg << "] — all targets are masked (-1), batch has nothing to train on";
        throw std::runtime_error(msg.str());
    }
    if (payload.lm_valid_tokens <= 0) {
        throw std::runtime_error(
            "buildBatchPayload: lm_valid_tokens=0 after execution-slot masking "
            "(valid_tokens=" + std::to_string(payload.valid_tokens) +
            ") — all LM targets were claimed by execution slots");
    }

    // Cross-check geometry invariants (Rule 20: crash if anything is wrong)
    payload.validate("buildBatchPayload");

    // Execution payload validation (WS4: single shared validator)
    GRIM::Execution::validateExecutionPayload(
        payload, "buildBatchPayload",
        execution_num_slots, execution_num_ops, execution_num_steps);

    return payload;
}

BatchPayload buildInferenceBatchPayload(
    const std::vector<int>& token_ids,
    const std::vector<float>& numeric_values,
    const std::vector<uint8_t>& atom_mask,
    const std::vector<uint32_t>& atom_flags,
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table,
    const std::vector<uint32_t>& atom_entry_ids,
    const std::vector<int32_t>& token_to_slot_map,
    int vocab_size,
    size_t batch_capacity,
    size_t max_cached_seq_len,
    int execution_num_slots,
    int number_encoder_digit_slots,
    int number_encoder_max_abs_pow10)
{
    const char* caller = "buildInferenceBatchPayload";
    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len <= 0) {
        throw std::runtime_error("buildInferenceBatchPayload: token_ids is empty");
    }
    if (static_cast<int>(numeric_values.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: numeric_values.size()=" +
            std::to_string(numeric_values.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }
    if (static_cast<int>(atom_mask.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: atom_mask.size()=" +
            std::to_string(atom_mask.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }
    if (static_cast<int>(atom_flags.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: atom_flags.size()=" +
            std::to_string(atom_flags.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }
    if (static_cast<int>(atom_entry_ids.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: atom_entry_ids.size()=" +
            std::to_string(atom_entry_ids.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }
    if (!token_to_slot_map.empty() && static_cast<int>(token_to_slot_map.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: token_to_slot_map.size()=" +
            std::to_string(token_to_slot_map.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }

    bool row_execution_active = false;
    if (!token_to_slot_map.empty()) {
        if (execution_num_slots <= 0) {
            throw std::runtime_error(
                "buildInferenceBatchPayload: execution_num_slots must be > 0 when token_to_slot_map is provided");
        }
        for (int t = 0; t < seq_len; ++t) {
            const int32_t slot = token_to_slot_map[static_cast<size_t>(t)];
            if (slot != -1) {
                if (slot < 0 || slot >= execution_num_slots) {
                    throw std::runtime_error(
                        "buildInferenceBatchPayload: token_to_slot_map[" + std::to_string(t) +
                        "]=" + std::to_string(slot) + " out of range [0, " +
                        std::to_string(execution_num_slots) + ") or -1");
                }
                row_execution_active = true;
            }
        }
    }

    for (int t = 0; t < seq_len; ++t) {
        const int token_id = token_ids[static_cast<size_t>(t)];
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error(
                "buildInferenceBatchPayload: token_ids[" + std::to_string(t) +
                "]=" + std::to_string(token_id) + " out of range [0, " +
                std::to_string(vocab_size) + ")");
        }
    }

    BatchPayload payload = makeInferenceBasePayload(
        seq_len, vocab_size, batch_capacity, max_cached_seq_len,
        BatchPayloadMode::InferencePrefill, row_execution_active, caller);

    payload.input_ids = token_ids;
    payload.target_ids.assign(static_cast<size_t>(seq_len), -1);
    payload.numeric_values = numeric_values;
    payload.atom_mask = atom_mask;
    payload.atom_flags = atom_flags;
    payload.atom_entry_ids = atom_entry_ids;
    payload.token_to_slot_map.assign(static_cast<size_t>(seq_len), -1);
    if (!token_to_slot_map.empty()) {
        payload.token_to_slot_map = token_to_slot_map;
    }
    payload.seq_atom_tables.resize(1);
    payload.seq_atom_tables[0] = atom_table;

    materializeAuthoredAtomFacts(payload, caller);
    materializeNumberEncoderChannels(
        payload, number_encoder_digit_slots, number_encoder_max_abs_pow10, caller);

    payload.validate(caller);
    return payload;
}

BatchPayload buildInferenceDecodePayload(int vocab_size)
{
    const char* caller = "buildInferenceDecodePayload";
    BatchPayload payload = makeInferenceBasePayload(
        1, vocab_size, 1, 1,
        BatchPayloadMode::InferenceDecode, true, caller);
    payload.validate(caller);
    return payload;
}

}  // namespace Batching
}  // namespace GRIM
