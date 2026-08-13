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

std::vector<int32_t> buildInferenceExecutionSlotIndexMap(
    const std::vector<int>& token_ids,
    const std::vector<uint8_t>& atom_mask,
    int num_slots,
    int num_scratch_slots)
{
    const char* caller = "buildInferenceExecutionSlotIndexMap";
    if (token_ids.size() != atom_mask.size()) {
        throw std::runtime_error(
            std::string(caller) + ": token_ids.size()=" + std::to_string(token_ids.size()) +
            " != atom_mask.size()=" + std::to_string(atom_mask.size()));
    }
    if (num_slots <= 0) {
        throw std::runtime_error(std::string(caller) + ": num_slots must be > 0");
    }
    if (num_scratch_slots < 0 || num_scratch_slots >= num_slots) {
        throw std::runtime_error(
            std::string(caller) + ": num_scratch_slots=" +
            std::to_string(num_scratch_slots) + " must be in [0, num_slots)");
    }

    std::vector<int32_t> slot_map(token_ids.size(), -1);
    int next_slot = num_scratch_slots;
    for (size_t t = 0; t < token_ids.size(); ++t) {
        if (atom_mask[t] == 0) {
            if (Tokenizer::isAtomTokenId(token_ids[t])) {
                throw std::runtime_error(
                    std::string(caller) + ": atom token at position " + std::to_string(t) +
                    " is missing tokenizer-authored atom_mask metadata");
            }
            continue;
        }
        if (!Tokenizer::isAtomTokenId(token_ids[t])) {
            throw std::runtime_error(
                std::string(caller) + ": atom_mask marks non-atom token at position " +
                std::to_string(t));
        }
        const auto atom_type = Tokenizer::tokenIdToAtomType(token_ids[t]);
        if (!Tokenizer::isNumericAtom(atom_type)) {
            throw std::runtime_error(
                std::string(caller) + ": non-numeric atom at position " +
                std::to_string(t) + " cannot bind ARG bootstrap value memory");
        }
        if (next_slot >= num_slots) {
            throw std::runtime_error(
                std::string(caller) + ": numeric atom count exceeds available value-slot capacity=" +
                std::to_string(num_slots - num_scratch_slots));
        }
        slot_map[t] = next_slot++;
    }
    return slot_map;
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
// Shared per-entry NumberEncoder feature fill. Used by BOTH the per-token input
// channels (materializeNumberEncoderChannels) and the candidate-pool key channels
// (materializeAtomEntryPool), so a candidate entry's selector key is encoded from
// the IDENTICAL features as the same number seen on the input side. Validates the
// numeric atom and writes digit/pow10/slot/global features at flat index `idx`
// (idx = per-token atom index or per-pool entry index). `err_context` is the
// caller-built message prefix used in all fail-loud throws.
void fillNumberEncoderEntryFeatures(
    const GRIM::Tokenizer::AtomEntry& entry,
    std::size_t idx,
    int digit_slots,
    int max_abs_pow10,
    std::vector<int>& digit_values,
    std::vector<int>& pow10_index,
    std::vector<float>& digit_mask,
    std::vector<float>& slot_features,
    std::vector<float>& global_features,
    const std::string& err_context)
{
    if (!GRIM::Tokenizer::isNumericAtom(entry.type)) {
        throw std::runtime_error(err_context + " is not a numeric atom type");
    }
    if (!entry.arg_number.has_value()) {
        throw std::runtime_error(err_context + " is missing required arg_number metadata");
    }
    const GRIM::Tokenizer::AtomNumber& number = *entry.arg_number;
    const std::size_t slots = static_cast<std::size_t>(digit_slots);
    const std::size_t digit_count = number.digits.size();
    if (digit_count == 0) {
        throw std::runtime_error(err_context + " has zero mantissa digit bindings");
    }
    if (digit_count > slots) {
        throw std::runtime_error(err_context + " has " + std::to_string(digit_count) +
            " mantissa digits exceeding number_encoder_max_digit_slots=" + std::to_string(digit_slots));
    }
    const float pow10_norm_scale = 1.0f / static_cast<float>(max_abs_pow10);
    const float digit_count_norm_scale = 1.0f / static_cast<float>(digit_slots);
    const float sign = number.sign_negative ? -1.0f : 1.0f;
    const bool is_float_atom = entry.type == GRIM::Tokenizer::AtomType::ATOM_FLOAT;
    const float is_float = is_float_atom ? 1.0f : 0.0f;
    const std::size_t slot_base = idx * slots;
    for (std::size_t i = 0; i < digit_count; ++i) {
        const auto& binding = number.digits[i];
        if (binding.digit > 9) {
            throw std::runtime_error(err_context + " digit[" + std::to_string(i) + "]=" +
                std::to_string(binding.digit) + " is not a base-10 digit");
        }
        const int pow10 = static_cast<int>(binding.pow10);
        if (pow10 < -max_abs_pow10 || pow10 > max_abs_pow10) {
            throw std::runtime_error(err_context + " digit[" + std::to_string(i) + "] pow10=" +
                std::to_string(pow10) + " outside configured number_encoder_max_abs_pow10=±" +
                std::to_string(max_abs_pow10));
        }
        const std::size_t slot = slot_base + i;
        digit_values[slot] = static_cast<int>(binding.digit);
        pow10_index[slot] = pow10 + max_abs_pow10;  // bucket [0, 2*max]
        digit_mask[slot] = 1.0f;
        float* feat = slot_features.data() +
                      slot * static_cast<std::size_t>(BatchPayload::kNumberSlotFeatureDim);
        feat[0] = static_cast<float>(binding.digit) / 9.0f;
        feat[1] = static_cast<float>(pow10) * pow10_norm_scale;
        feat[2] = binding.digit == 0 ? 1.0f : 0.0f;
        feat[3] = sign;
        feat[4] = is_float;
    }
    float* gfeat = global_features.data() +
                   idx * static_cast<std::size_t>(BatchPayload::kNumberGlobalFeatureDim);
    gfeat[0] = sign;
    gfeat[1] = static_cast<float>(number.exponent_value) * pow10_norm_scale;
    gfeat[2] = static_cast<float>(number.integer_digit_count) * digit_count_norm_scale;
    gfeat[3] = static_cast<float>(number.fractional_digit_count) * digit_count_norm_scale;
    gfeat[4] = number.has_decimal_point ? 1.0f : 0.0f;
    gfeat[5] = is_float;
}

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
        fillNumberEncoderEntryFeatures(
            *entry, atom_idx, digit_slots, max_abs_pow10,
            payload.atom_digit_values, payload.atom_digit_pow10_index,
            payload.atom_digit_mask, payload.atom_digit_slot_features,
            payload.atom_global_features,
            std::string(caller) + ": number-encoder atom_entry_id=" +
            std::to_string(entry_id) + " at token_pos=" + std::to_string(token_pos));
    }
}

// =============================================================================
// materializeAtomEntryPool — candidate "menu" for the arg/option selector.
//
// Merges every batch row's AtomTable entries into one batch-global pool with
// per-row windows (row_atom_offset), so the device selector can score candidate
// options. Execution-INDEPENDENT: the pool is built purely from authored
// AtomTable data, never from ExecutionMemory (GRIM/Docs/DeletedCode.md).
//
// Gated only on selector_enabled. Candidate metadata does not depend on
// NumberEncoder construction. Must be called after atom side-channels and
// seq_atom_tables are assembled.
//
// Each row r enumerates its entries in id order 0..size-1, so a token's batch-
// global pool index is row_atom_offset[row] + <row-local atom_entry_id>. A
// fail-loud agreement check confirms the exact pool agrees with the legacy float
// lane (numeric_values) and the token's atom type before any consumer relies on
// it (transitional, per docs/ATOM_PLACEHOLDER_SELECTOR_INFERENCE_PLAN.md Phase 1).
// =============================================================================
void materializeAtomEntryPool(
    BatchPayload& payload,
    bool selector_enabled,
    const char* caller)
{
    payload.num_pool_atoms = 0;
    payload.row_atom_offset.clear();
    payload.pool_numeric_values.clear();
    payload.pool_numeric_float_values.clear();
    payload.pool_numeric_int_values.clear();
    payload.pool_numeric_kinds.clear();
    payload.pool_atom_types.clear();
    payload.arg_select_targets.clear();
    payload.arg_select_valid_count = 0;

    if (!selector_enabled) {
        return;
    }
    if (!payload.ownsHostInputData()) {
        throw std::runtime_error(
            std::string(caller) + ": atom-entry pool requested for a payload mode (" +
            payload.modeName() + ") that owns no host input data");
    }
    if (static_cast<int>(payload.seq_atom_tables.size()) != payload.batch_size) {
        throw std::runtime_error(
            std::string(caller) + ": seq_atom_tables.size()=" +
            std::to_string(payload.seq_atom_tables.size()) + " != batch_size=" +
            std::to_string(payload.batch_size) + " while building the atom-entry pool");
    }

    // Pass 1: per-row windows + total entry count.
    payload.row_atom_offset.assign(static_cast<std::size_t>(payload.batch_size) + 1, 0);
    for (int row = 0; row < payload.batch_size; ++row) {
        const auto& atom_table = payload.seq_atom_tables[static_cast<std::size_t>(row)];
        const int row_entries = atom_table ? static_cast<int>(atom_table->size()) : 0;
        payload.row_atom_offset[static_cast<std::size_t>(row) + 1] =
            payload.row_atom_offset[static_cast<std::size_t>(row)] + row_entries;
    }
    const int total_entries = payload.row_atom_offset.back();
    payload.num_pool_atoms = total_entries;

    const std::size_t E = static_cast<std::size_t>(total_entries);
    payload.pool_numeric_values.assign(E, 0.0f);
    payload.pool_numeric_float_values.assign(E, 0.0);
    payload.pool_numeric_int_values.assign(E, 0);
    payload.pool_numeric_kinds.assign(
        E, static_cast<uint8_t>(GRIM::Tokenizer::NumericPayloadKind::NONE));
    payload.pool_atom_types.assign(E, 0);
    // Pass 2: exact per-entry value and type metadata.
    for (int row = 0; row < payload.batch_size; ++row) {
        const auto& atom_table = payload.seq_atom_tables[static_cast<std::size_t>(row)];
        const int row_entries = atom_table ? static_cast<int>(atom_table->size()) : 0;
        const int base = payload.row_atom_offset[static_cast<std::size_t>(row)];
        for (uint32_t entry_id = 0; entry_id < static_cast<uint32_t>(row_entries); ++entry_id) {
            const auto entry = atom_table->getAtom(entry_id);
            if (!entry.has_value()) {
                throw std::runtime_error(
                    std::string(caller) + ": atom-entry pool row " + std::to_string(row) +
                    " entry id " + std::to_string(entry_id) +
                    " not retrievable (entry ids must be contiguous 0..size-1)");
            }
            const std::size_t pool_idx = static_cast<std::size_t>(base) + entry_id;
            const auto numeric_payload = atom_table->getNumericValue(entry_id);
            if (!numeric_payload.has_value()) {
                throw std::runtime_error(
                    std::string(caller) + ": atom-entry pool row " + std::to_string(row) +
                    " entry id " + std::to_string(entry_id) +
                    " has no exact numeric payload");
            }
            payload.pool_numeric_values[pool_idx] = entry->numeric_value;
            payload.pool_numeric_float_values[pool_idx] = numeric_payload->float_value;
            payload.pool_numeric_int_values[pool_idx] = numeric_payload->int_value;
            payload.pool_numeric_kinds[pool_idx] =
                static_cast<uint8_t>(numeric_payload->kind);
            payload.pool_atom_types[pool_idx] = static_cast<int>(entry->type);
        }
    }

    // Fail-loud agreement: every atom token's batch-global pool index must resolve
    // to the same value (float-exact, same source) and atom type.
    for (int token_pos = 0; token_pos < payload.total_tokens; ++token_pos) {
        if (payload.atom_mask[static_cast<std::size_t>(token_pos)] == 0) {
            continue;
        }
        const int row = token_pos / payload.max_seq_len;
        if (row < 0 || row >= payload.batch_size) {
            throw std::runtime_error(
                std::string(caller) + ": atom token_pos=" + std::to_string(token_pos) +
                " resolves to row " + std::to_string(row) + " out of range");
        }
        const uint32_t entry_id = payload.atom_entry_ids[static_cast<std::size_t>(token_pos)];
        const int pool_index = payload.row_atom_offset[static_cast<std::size_t>(row)] +
                               static_cast<int>(entry_id);
        if (entry_id == GRIM::Tokenizer::kAtomEntryNone ||
            pool_index < payload.row_atom_offset[static_cast<std::size_t>(row)] ||
            pool_index >= payload.row_atom_offset[static_cast<std::size_t>(row) + 1]) {
            throw std::runtime_error(
                std::string(caller) + ": atom token_pos=" + std::to_string(token_pos) +
                " entry_id=" + std::to_string(entry_id) +
                " falls outside row " + std::to_string(row) + " pool window [" +
                std::to_string(payload.row_atom_offset[static_cast<std::size_t>(row)]) + ", " +
                std::to_string(payload.row_atom_offset[static_cast<std::size_t>(row) + 1]) + ")");
        }
        if (payload.pool_numeric_values[static_cast<std::size_t>(pool_index)] !=
            payload.numeric_values[static_cast<std::size_t>(token_pos)]) {
            throw std::runtime_error(
                std::string(caller) + ": atom-entry pool value disagreement at token_pos=" +
                std::to_string(token_pos) + " (pool=" +
                std::to_string(payload.pool_numeric_values[static_cast<std::size_t>(pool_index)]) +
                " vs numeric_values=" +
                std::to_string(payload.numeric_values[static_cast<std::size_t>(token_pos)]) + ")");
        }
        const int token_type = static_cast<int>(
            GRIM::Tokenizer::tokenIdToAtomType(payload.input_ids[static_cast<std::size_t>(token_pos)]));
        if (payload.pool_atom_types[static_cast<std::size_t>(pool_index)] != token_type) {
            throw std::runtime_error(
                std::string(caller) + ": atom-entry pool type disagreement at token_pos=" +
                std::to_string(token_pos));
        }
    }

    // ── Arg/option selector supervision: next-atom-entry targets. ───────────────
    // At position t, the target is the batch-global pool index of token t+1's atom
    // entry (when t+1 is an atom in the same row): "select the option that the next
    // token turns out to be." Supervision only — never fed back as input at t.
    payload.arg_select_targets.assign(static_cast<std::size_t>(payload.total_tokens), -1);
    payload.arg_select_valid_count = 0;
    for (int token_pos = 0; token_pos + 1 < payload.total_tokens; ++token_pos) {
        const int next = token_pos + 1;
        const int row = token_pos / payload.max_seq_len;
        if (next / payload.max_seq_len != row) {
            continue;  // next token belongs to a different batch row
        }
        if (payload.atom_mask[static_cast<std::size_t>(next)] == 0) {
            continue;  // next token is not a selectable atom
        }
        const uint32_t local = payload.atom_entry_ids[static_cast<std::size_t>(next)];
        if (local == GRIM::Tokenizer::kAtomEntryNone) {
            continue;
        }
        const int global_idx =
            payload.row_atom_offset[static_cast<std::size_t>(row)] + static_cast<int>(local);
        if (global_idx < payload.row_atom_offset[static_cast<std::size_t>(row)] ||
            global_idx >= payload.row_atom_offset[static_cast<std::size_t>(row) + 1]) {
            throw std::runtime_error(
                std::string(caller) + ": arg-select target for token_pos=" +
                std::to_string(token_pos) + " resolves outside row " + std::to_string(row) + " window");
        }
        payload.arg_select_targets[static_cast<std::size_t>(token_pos)] = global_idx;
        ++payload.arg_select_valid_count;
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
    bool selector_enabled,
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

    materializeAuthoredAtomFacts(payload, "buildBatchPayload");
    materializeNumberEncoderChannels(
        payload, number_encoder_digit_slots, number_encoder_max_abs_pow10, "buildBatchPayload");
    materializeAtomEntryPool(payload, selector_enabled, "buildBatchPayload");

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
    bool selector_enabled,
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

    materializeAuthoredAtomFacts(payload, caller);
    materializeNumberEncoderChannels(
        payload, number_encoder_digit_slots, number_encoder_max_abs_pow10, caller);
    materializeAtomEntryPool(payload, selector_enabled, caller);
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
