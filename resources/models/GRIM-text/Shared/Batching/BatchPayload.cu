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
#include "../Goal/Goal.hpp"
#include "../Goal/GoalSpanView.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"
#include "../TokenizerArtifacts/GrmtSequence.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
namespace GRIM {
namespace Batching {

GRIM::GoalSpanView BatchPayload::goalSpansForRow(std::size_t row) const {
    if (batch_size <= 0) {
        throw std::runtime_error(
            "BatchPayload::goalSpansForRow: batch_size must be > 0");
    }
    if (row >= static_cast<std::size_t>(batch_size)) {
        throw std::out_of_range(
            "BatchPayload::goalSpansForRow: row=" + std::to_string(row) +
            " is outside batch_size=" + std::to_string(batch_size));
    }
    if (goals.empty()) {
        if (isTraining()) {
            throw std::runtime_error(
                "BatchPayload::goalSpansForRow: training goals array is empty");
        }
        return GRIM::GoalSpanView{};
    }
    if (goals.size() != static_cast<std::size_t>(batch_size)) {
        throw std::runtime_error(
            "BatchPayload::goalSpansForRow: goals.size()=" +
            std::to_string(goals.size()) + " != batch_size=" +
            std::to_string(batch_size));
    }
    const GRIM::Goal* goal = goals[row].get();
    if (!goal) {
        return GRIM::GoalSpanView{};
    }
    const GRIM::GoalTokenSpan* target_state = goal->target_state.has_value()
        ? &goal->target_state->span
        : nullptr;
    const GRIM::SuccessCriteria* success_criteria =
        goal->success_criteria.has_value()
            ? &*goal->success_criteria
            : nullptr;
    const GRIM::Constraints* constraints = goal->constraints.has_value()
        ? &*goal->constraints
        : nullptr;
    return GRIM::GoalSpanView(target_state, success_criteria, constraints);
}

const uint8_t* BatchPayload::atomAuxTargetMaskForRow(std::size_t row) const {
    if (row >= static_cast<std::size_t>(batch_size)) {
        throw std::out_of_range(
            "BatchPayload::atomAuxTargetMaskForRow: row=" +
            std::to_string(row) + " is outside batch_size=" +
            std::to_string(batch_size));
    }
    if (atom_aux_target_mask.size() != static_cast<std::size_t>(total_tokens)) {
        throw std::runtime_error(
            "BatchPayload::atomAuxTargetMaskForRow: atom_aux_target_mask.size()=" +
            std::to_string(atom_aux_target_mask.size()) + " != total_tokens=" +
            std::to_string(total_tokens));
    }
    return atom_aux_target_mask.data() + row * static_cast<std::size_t>(max_seq_len);
}

namespace {

void requirePositiveVocab(int vocab_size, const char* caller)
{
    if (vocab_size <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": vocab_size=" + std::to_string(vocab_size) +
            " (must be > 0)");
    }
}

void materializeCompactAtomOpenings(BatchPayload& payload, const char* caller)
{
    payload.atom_positions.clear();
    payload.atom_types.clear();
    if (!payload.ownsHostInputData()) {
        return;
    }
    if (payload.input_ids.size() != static_cast<std::size_t>(payload.total_tokens) ||
        payload.atom_mask.size() != static_cast<std::size_t>(payload.total_tokens)) {
        throw std::runtime_error(
            std::string(caller) +
            ": cannot materialize compact atom openings before input_ids/atom_mask are complete");
    }

    for (int row = 0; row < payload.batch_size; ++row) {
        const int row_length = payload.seq_lengths[static_cast<std::size_t>(row)];
        const int row_offset = row * payload.max_seq_len;
        for (int token = 0; token < row_length; ++token) {
            const int flat_position = row_offset + token;
            const int token_id = payload.input_ids[static_cast<std::size_t>(flat_position)];
            if (!GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
                continue;
            }
            if (payload.atom_mask[static_cast<std::size_t>(flat_position)] != 1) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": typed atom opening is missing atom_mask=1 at flat position=" +
                    std::to_string(flat_position));
            }
            payload.atom_positions.push_back(flat_position);
            payload.atom_types.push_back(static_cast<int>(
                GRIM::Tokenizer::tokenIdToAtomType(token_id)));
        }
    }
}

void materializeNumberAuxTargets(
    BatchPayload& payload,
    int digit_slots,
    int max_abs_pow10,
    const char* caller)
{
    payload.number_aux_target_digit_slots = 0;
    payload.number_aux_target_max_abs_pow10 = 0;
    payload.number_aux_target_valid_count = 0;
    payload.number_aux_target_valid_row_count = 0;
    payload.number_aux_target_atom_index.clear();
    payload.number_aux_target_row_mask.clear();
    payload.number_aux_target_step_index.clear();
    payload.number_aux_target_valid.clear();
    payload.number_aux_target_sign_negative.clear();
    payload.number_aux_target_base.clear();
    payload.number_aux_target_digit_count.clear();
    payload.number_aux_target_digits.clear();
    payload.number_aux_target_pow10_index.clear();
    payload.number_aux_target_digit_mask.clear();
    payload.number_aux_target_float_values.clear();
    payload.number_aux_target_int_values.clear();
    payload.number_aux_target_numeric_kinds.clear();

    if (digit_slots == 0) {
        if (max_abs_pow10 != 0) {
            throw std::runtime_error(
                std::string(caller) +
                ": number auxiliary max_abs_pow10 must be 0 when digit_slots=0");
        }
        return;
    }
    if (digit_slots < 0 || max_abs_pow10 < 0) {
        throw std::runtime_error(
            std::string(caller) +
            ": number auxiliary target geometry must be non-negative");
    }

    const std::size_t atoms = payload.atom_positions.size();
    const std::size_t slots = static_cast<std::size_t>(digit_slots);
    payload.number_aux_target_digit_slots = digit_slots;
    payload.number_aux_target_max_abs_pow10 = max_abs_pow10;
    payload.number_aux_target_atom_index.assign(
        static_cast<std::size_t>(payload.total_tokens), -1);
    payload.number_aux_target_row_mask.assign(
        static_cast<std::size_t>(payload.total_tokens), 0);
    payload.number_aux_target_step_index.assign(
        static_cast<std::size_t>(payload.total_tokens), -1);
    payload.number_aux_target_valid.assign(atoms, 0);
    payload.number_aux_target_sign_negative.assign(atoms, 0);
    payload.number_aux_target_base.assign(atoms, 0);
    payload.number_aux_target_digit_count.assign(atoms, 0);
    payload.number_aux_target_digits.assign(atoms * slots, 0);
    payload.number_aux_target_pow10_index.assign(atoms * slots, 0);
    payload.number_aux_target_digit_mask.assign(atoms * slots, 0);
    payload.number_aux_target_float_values.assign(atoms, 0.0);
    payload.number_aux_target_int_values.assign(atoms, 0);
    payload.number_aux_target_numeric_kinds.assign(
        atoms, static_cast<uint8_t>(GRIM::Tokenizer::NumericPayloadKind::NONE));

    for (std::size_t atom = 0; atom < atoms; ++atom) {
        const auto atom_type = static_cast<GRIM::Tokenizer::AtomType>(
            payload.atom_types[atom]);
        if (!GRIM::Tokenizer::isNumericAtom(atom_type)) {
            continue;
        }

        const int flat_position = payload.atom_positions[atom];
        const int row = flat_position / payload.max_seq_len;
        if (row < 0 || row >= payload.batch_size ||
            static_cast<std::size_t>(row) >= payload.seq_atom_tables.size() ||
            !payload.seq_atom_tables[static_cast<std::size_t>(row)]) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric opening cannot resolve its row AtomTable at atom=" +
                std::to_string(atom));
        }

        const uint32_t entry_id =
            payload.atom_entry_ids[static_cast<std::size_t>(flat_position)];
        if (entry_id == GRIM::Tokenizer::kAtomEntryNone) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric opening has no atom_entry_id at flat position=" +
                std::to_string(flat_position));
        }
        const auto& table = payload.seq_atom_tables[static_cast<std::size_t>(row)];
        const auto entry = table->getAtom(entry_id);
        if (!entry.has_value()) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric opening references missing AtomEntry id=" +
                std::to_string(entry_id));
        }
        if (entry->type != atom_type || !entry->arg_number.has_value()) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric AtomEntry type/decomposition mismatch for id=" +
                std::to_string(entry_id));
        }

        const auto& number = *entry->arg_number;
        if (number.digits.empty() || number.digits.size() > slots) {
            throw std::runtime_error(
                std::string(caller) + ": numeric AtomEntry id=" +
                std::to_string(entry_id) + " has digit_count=" +
                std::to_string(number.digits.size()) +
                " outside configured digit_slots=" + std::to_string(digit_slots));
        }
        if (number.base != 10) {
            throw std::runtime_error(
                std::string(caller) + ": numeric AtomEntry id=" +
                std::to_string(entry_id) + " has unsupported base=" +
                std::to_string(number.base));
        }

        payload.number_aux_target_sign_negative[atom] = number.sign_negative;
        payload.number_aux_target_base[atom] = number.base;
        payload.number_aux_target_digit_count[atom] =
            static_cast<uint16_t>(number.digits.size());
        for (std::size_t slot = 0; slot < number.digits.size(); ++slot) {
            const auto& digit = number.digits[slot];
            const int pow10 = static_cast<int>(digit.pow10);
            if (pow10 < -max_abs_pow10 || pow10 > max_abs_pow10) {
                throw std::runtime_error(
                    std::string(caller) + ": numeric AtomEntry id=" +
                    std::to_string(entry_id) + " has pow10=" +
                    std::to_string(pow10) + " outside configured range +/-" +
                    std::to_string(max_abs_pow10));
            }
            const std::size_t index = atom * slots + slot;
            payload.number_aux_target_digits[index] = static_cast<int>(digit.digit);
            payload.number_aux_target_pow10_index[index] = pow10 + max_abs_pow10;
            payload.number_aux_target_digit_mask[index] = 1;
        }

        const auto numeric = table->getNumericValue(entry_id);
        if (!numeric.has_value()) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric AtomEntry has no exact decoded value for id=" +
                std::to_string(entry_id));
        }
        const auto expected_kind = atom_type == GRIM::Tokenizer::AtomType::ATOM_INT
            ? GRIM::Tokenizer::NumericPayloadKind::INTEGER
            : GRIM::Tokenizer::NumericPayloadKind::FLOAT;
        if (numeric->kind != expected_kind) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric AtomEntry exact-value kind disagrees with atom type for id=" +
                std::to_string(entry_id));
        }
        payload.number_aux_target_float_values[atom] = numeric->float_value;
        payload.number_aux_target_int_values[atom] = numeric->int_value;
        payload.number_aux_target_numeric_kinds[atom] =
            static_cast<uint8_t>(numeric->kind);

        const int row_end = row * payload.max_seq_len +
            payload.seq_lengths[static_cast<std::size_t>(row)];
        bool any_valid_row = false;
        int position = flat_position;
        int decoder_step = 0;
        for (; position < row_end &&
               payload.atom_aux_target_mask[static_cast<std::size_t>(position)] != 0;
             ++position, ++decoder_step) {
            payload.number_aux_target_atom_index[static_cast<std::size_t>(position)] =
                static_cast<int>(atom);
            payload.number_aux_target_step_index[static_cast<std::size_t>(position)] =
                decoder_step;
            if (payload.target_ids[static_cast<std::size_t>(position)] >= 0) {
                payload.number_aux_target_row_mask[static_cast<std::size_t>(position)] = 1;
                ++payload.number_aux_target_valid_row_count;
                any_valid_row = true;
            }
        }
        if (position >= row_end ||
            !GRIM::Tokenizer::isAtomCloseTokenId(
                payload.input_ids[static_cast<std::size_t>(position)]) ||
            GRIM::Tokenizer::tokenIdToAtomType(
                payload.input_ids[static_cast<std::size_t>(position)]) != atom_type) {
            throw std::runtime_error(
                std::string(caller) +
                ": numeric auxiliary span does not terminate at its matching close for AtomEntry id=" +
                std::to_string(entry_id));
        }
        if (any_valid_row) {
            payload.number_aux_target_valid[atom] = 1;
            ++payload.number_aux_target_valid_count;
        }
    }
}

BatchPayload makeInferenceBasePayload(
    int seq_len,
    int vocab_size,
    size_t batch_capacity,
    size_t max_cached_seq_len,
    BatchPayloadMode mode,
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

    return payload;
}

}  // namespace

BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout&,
    size_t batch_size,
    size_t max_cached_seq_len,
    bool,
    int number_aux_target_digit_slots,
    int number_aux_target_max_abs_pow10)
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
        const std::vector<uint8_t>* atom_aux_target_mask;
        const std::vector<uint32_t>* atom_flags;
        std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table;
        const std::vector<uint32_t>* atom_entry_ids;
        const std::vector<int32_t>* exec_slots;
        int length;

    };

    std::vector<RawSeq> raw;
    raw.reserve(payload.batch_size);

    payload.seq_lengths.resize(payload.batch_size);
    payload.prompt_lengths.resize(payload.batch_size, 0);
    payload.prompt_end_positions.resize(payload.batch_size, -1);
    payload.goals.resize(payload.batch_size);
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
        if (static_cast<int>(seq->token_atom_aux_target_mask.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " token_atom_aux_target_mask.size()=" +
                std::to_string(seq->token_atom_aux_target_mask.size()) +
                " != token_ids.size()=" + std::to_string(seq_len) +
                " - sliding-window construction must author auxiliary ownership");
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

        }

        const std::vector<int32_t>* exec_slots_ptr = nullptr;
        if (!seq->token_exec_slot_indices.empty()) {
            if (static_cast<int>(seq->token_exec_slot_indices.size()) != seq_len) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " token_exec_slot_indices.size()=" + std::to_string(seq->token_exec_slot_indices.size()) +
                    " != token_ids.size()=" + std::to_string(seq_len));
            }
            exec_slots_ptr = &seq->token_exec_slot_indices;
        }

        raw.push_back({
            &seq->token_ids,
            &seq->targets,
            &seq->token_numeric_values,
            &seq->token_atom_mask,
            &seq->token_atom_aux_target_mask,
            &seq->token_atom_flags,
            seq->atom_table,
            &seq->atom_entry_ids,
            exec_slots_ptr,
            seq_len
        });

        payload.seq_lengths[b] = seq_len;
        payload.prompt_lengths[b] = seq->prompt_length;
        payload.prompt_end_positions[b] = seq->prompt_end_pos;
        payload.goals[b] = seq->goal;
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
    payload.atom_aux_target_mask.assign(flat_size, 0);
    payload.atom_flags.assign(flat_size, 0);
    payload.atom_entry_ids.assign(flat_size, GRIM::Tokenizer::kAtomEntryNone);
    payload.token_to_slot_index_map.assign(flat_size, -1);
    payload.seq_atom_tables.resize(payload.batch_size);
    payload.valid_target_counts.resize(payload.batch_size, 0);

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

        // Copy causal auxiliary-head ownership authored by sliding windows.
        std::memcpy(&payload.atom_aux_target_mask[row_offset],
                    r.atom_aux_target_mask->data(),
                    seq_len * sizeof(uint8_t));

        // Bulk copy atom flags (type-specific metadata from AtomTable)
        std::memcpy(&payload.atom_flags[row_offset],
                    r.atom_flags->data(),
                    seq_len * sizeof(uint32_t));

        // Copy atom entry IDs (bulk memcpy — fixed-size uint32_t)
        std::memcpy(&payload.atom_entry_ids[row_offset],
                    r.atom_entry_ids->data(),
                    seq_len * sizeof(uint32_t));

        // Copy ARG bootstrap slot map (-1 for tokens that do not seed a slot).
        if (r.exec_slots) {
            std::memcpy(&payload.token_to_slot_index_map[row_offset],
                        r.exec_slots->data(),
                        seq_len * sizeof(int32_t));
        }
        // else: stays -1 (default) — dataset doesn't yet provide slot assignments

        // Store AtomTable reference for this batch row
        payload.seq_atom_tables[b] = r.atom_table;

    }

    materializeCompactAtomOpenings(payload, "buildBatchPayload");
    materializeNumberAuxTargets(
        payload,
        number_aux_target_digit_slots,
        number_aux_target_max_abs_pow10,
        "buildBatchPayload");

    // ═════════════════════════════════════════════════════════════════════════
    // ARG bootstrap metadata does not alter LM target ownership.
    // ═════════════════════════════════════════════════════════════════════════
    payload.lm_valid_tokens = payload.valid_tokens;

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
            "buildBatchPayload: lm_valid_tokens=0 after target masking "
            "(valid_tokens=" + std::to_string(payload.valid_tokens) + ")");
    }

    // Cross-check geometry invariants (Rule 20: crash if anything is wrong)
    payload.validate("buildBatchPayload");

    return payload;
}

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
    bool,
    int,
    int)
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
    if (!token_to_slot_index_map.empty() && static_cast<int>(token_to_slot_index_map.size()) != seq_len) {
        throw std::runtime_error(
            "buildInferenceBatchPayload: token_to_slot_index_map.size()=" +
            std::to_string(token_to_slot_index_map.size()) + " != token_ids.size()=" +
            std::to_string(seq_len));
    }

    // A full-length all--1 map is the canonical "no authored bindings" form.
    // Execution geometry is required only when at least one token actually
    // names a slot; vector presence alone does not make execution active.
    bool has_slot_bindings = false;
    for (std::size_t t = 0; t < token_to_slot_index_map.size(); ++t) {
        const int32_t slot = token_to_slot_index_map[t];
        if (slot < -1) {
            throw std::runtime_error(
                "buildInferenceBatchPayload: token_to_slot_index_map[" +
                std::to_string(t) + "]=" + std::to_string(slot) +
                " must be -1 or non-negative");
        }
        has_slot_bindings = has_slot_bindings || slot >= 0;
    }

    // Non-negative entries supply authored ARG bootstrap bindings.
    if (has_slot_bindings) {
        if (execution_num_slots <= 0) {
            throw std::runtime_error(
                "buildInferenceBatchPayload: execution_num_slots must be > 0 when an authored slot binding exists");
        }
        if (execution_num_scratch_slots < 0 ||
            execution_num_scratch_slots >= execution_num_slots) {
            throw std::runtime_error(
                "buildInferenceBatchPayload: execution_num_scratch_slots=" +
                std::to_string(execution_num_scratch_slots) +
                " must be in [0, execution_num_slots)");
        }
        std::vector<int> slot_token_positions(
            static_cast<std::size_t>(execution_num_slots), -1);
        for (int t = 0; t < seq_len; ++t) {
            const int32_t slot = token_to_slot_index_map[static_cast<size_t>(t)];
            if (slot != -1) {
                if (slot < execution_num_scratch_slots || slot >= execution_num_slots) {
                    throw std::runtime_error(
                        "buildInferenceBatchPayload: token_to_slot_index_map[" + std::to_string(t) +
                        "]=" + std::to_string(slot) + " out of value-slot range [" +
                        std::to_string(execution_num_scratch_slots) + ", " +
                        std::to_string(execution_num_slots) + ") or -1");
                }
                const int prior_token_pos =
                    slot_token_positions[static_cast<std::size_t>(slot)];
                if (prior_token_pos >= 0) {
                    throw std::runtime_error(
                        "buildInferenceBatchPayload: token positions " +
                        std::to_string(prior_token_pos) + " and " +
                        std::to_string(t) + " map to duplicate bootstrap slot " +
                        std::to_string(slot));
                }
                slot_token_positions[static_cast<std::size_t>(slot)] = t;
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
        BatchPayloadMode::InferencePrefill, caller);

    payload.input_ids = token_ids;
    payload.target_ids.assign(static_cast<size_t>(seq_len), -1);
    payload.numeric_values = numeric_values;
    payload.atom_mask = atom_mask;
    payload.atom_flags = atom_flags;
    payload.atom_entry_ids = atom_entry_ids;
    payload.token_to_slot_index_map.assign(static_cast<size_t>(seq_len), -1);
    if (!token_to_slot_index_map.empty()) {
        payload.token_to_slot_index_map = token_to_slot_index_map;
    }
    payload.seq_atom_tables.resize(1);
    payload.seq_atom_tables[0] = atom_table;

    materializeCompactAtomOpenings(payload, caller);
    payload.validate(caller);
    return payload;
}

BatchPayload buildInferenceDecodePayload(int vocab_size)
{
    const char* caller = "buildInferenceDecodePayload";
    BatchPayload payload = makeInferenceBasePayload(
        1, vocab_size, 1, 1,
        BatchPayloadMode::InferenceDecode, caller);
    payload.validate(caller);
    return payload;
}

}  // namespace Batching
}  // namespace GRIM
