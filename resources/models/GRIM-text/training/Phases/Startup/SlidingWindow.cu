//======================================================//
//  Startup/SlidingWindow.cu
//
//  Implementation of applySlidingWindows. Pure CPU code, but lives
//  in a .cu so the train_gpu translation unit list stays uniform
//  (mirrors Startup/Logging.cu and Startup/Rng.cu).
//======================================================//

#include "SlidingWindow.hpp"
#include "../../../Shared/ConceptBlock/ConceptBlockSpans.hpp"
#include "../../../Shared/Goal/Goal.hpp"
#include "../../../Shared/UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIMText::Training {

namespace {

using GrmtSequence = GRIM::TokenizerArtifacts::GrmtSequence;

struct SftWindowConstruction {
    std::vector<GrmtSequence> sequences;
    size_t long_sequence_count = 0;
    size_t generated_window_count = 0;
    size_t atom_safe_cut_adjustments = 0;
};

struct AtomTokenSpan {
    size_t begin = 0;
    size_t end = 0;  // exclusive
};

std::vector<AtomTokenSpan> collectAtomTokenSpans(
    const std::vector<int>& token_ids,
    const std::string& source) {
    std::vector<AtomTokenSpan> spans;
    spans.reserve(token_ids.size() / 8);

    size_t open_index = 0;
    std::optional<GRIM::Tokenizer::AtomType> open_type;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        const int token_id = token_ids[i];
        if (GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
            if (open_type.has_value()) {
                throw std::runtime_error(
                    source + ": nested typed atom opening at token index=" +
                    std::to_string(i) + " inside span opened at index=" +
                    std::to_string(open_index));
            }
            open_index = i;
            open_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
            continue;
        }
        if (!GRIM::Tokenizer::isAtomCloseTokenId(token_id)) {
            continue;
        }
        if (!open_type.has_value()) {
            throw std::runtime_error(
                source + ": typed atom closing boundary has no opening at token index=" +
                std::to_string(i));
        }
        const auto close_type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        if (close_type != *open_type) {
            throw std::runtime_error(
                source + ": typed atom boundary type mismatch: opening=" +
                std::string(GRIM::Tokenizer::atomTypeName(*open_type)) +
                " at index=" + std::to_string(open_index) + " closing=" +
                GRIM::Tokenizer::atomTypeName(close_type) + " at index=" +
                std::to_string(i));
        }
        spans.push_back(AtomTokenSpan{open_index, i + 1});
        open_type.reset();
    }

    if (open_type.has_value()) {
        throw std::runtime_error(
            source + ": typed atom opening boundary has no closing boundary at token index=" +
            std::to_string(open_index));
    }
    return spans;
}

void authorAtomAuxTargetMask(
    GrmtSequence& sequence,
    const std::vector<AtomTokenSpan>& spans,
    const std::string& source) {
    sequence.token_atom_aux_target_mask.assign(sequence.token_ids.size(), 0);
    for (const AtomTokenSpan& span : spans) {
        if (span.begin >= span.end || span.end > sequence.token_ids.size()) {
            throw std::runtime_error(
                source + ": invalid typed atom token span [" +
                std::to_string(span.begin) + "," + std::to_string(span.end) +
                ") for sequence length=" +
                std::to_string(sequence.token_ids.size()));
        }

        // AtomTokenSpan::end is one past the close delimiter. Causal rows from
        // the opening boundary up to (but excluding) the close-boundary row
        // predict every value token and finally the close delimiter itself.
        const size_t close_position = span.end - 1;
        std::fill(
            sequence.token_atom_aux_target_mask.begin() +
                static_cast<ptrdiff_t>(span.begin),
            sequence.token_atom_aux_target_mask.begin() +
                static_cast<ptrdiff_t>(close_position),
            static_cast<uint8_t>(1));
    }
}

void rebaseSequenceLocalAtoms(
    GrmtSequence& sequence,
    const std::shared_ptr<const GRIM::Tokenizer::SequenceLocalAtomTable>& source_table,
    const std::string& source) {
    if (sequence.token_local_atom_indices.size() != sequence.token_ids.size()) {
        throw std::runtime_error(
            source + ": token_local_atom_indices.size()=" +
            std::to_string(sequence.token_local_atom_indices.size()) +
            " != token_ids.size()=" + std::to_string(sequence.token_ids.size()));
    }

    auto rebased = std::make_shared<GRIM::Tokenizer::SequenceLocalAtomTable>();
    for (size_t i = 0; i < sequence.token_ids.size(); ++i) {
        const int token_id = sequence.token_ids[i];
        uint32_t& local_index = sequence.token_local_atom_indices[i];
        if (!GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
            if (local_index != GRIM::Tokenizer::kLocalAtomIndexNone) {
                throw std::runtime_error(
                    source + ": local atom index is present outside an opening boundary at token index=" +
                    std::to_string(i));
            }
            continue;
        }

        const auto type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        if (!source_table) {
            throw std::runtime_error(
                source + ": atom opening has no source sequence-local atom table at token index=" +
                std::to_string(i));
        }
        const auto raw_text = source_table->getRawText(type, local_index);
        if (!raw_text.has_value()) {
            throw std::runtime_error(
                source + ": cannot resolve source local atom address (" +
                std::string(GRIM::Tokenizer::atomTypeName(type)) + ", " +
                std::to_string(local_index) + ") at token index=" +
                std::to_string(i));
        }
        local_index = rebased->ticket(type, *raw_text).local_index;
    }
    sequence.local_atom_table = std::move(rebased);
}

const AtomTokenSpan* atomSpanContainingCut(
    const std::vector<AtomTokenSpan>& spans,
    size_t cut) {
    const auto after = std::upper_bound(
        spans.begin(), spans.end(), cut,
        [](size_t value, const AtomTokenSpan& span) {
            return value < span.begin;
        });
    if (after == spans.begin()) {
        return nullptr;
    }
    const AtomTokenSpan& candidate = *(after - 1);
    return candidate.begin < cut && cut < candidate.end ? &candidate : nullptr;
}

size_t rewindCutToAtomOpening(
    const std::vector<AtomTokenSpan>& spans,
    size_t cut,
    size_t& adjustment_count) {
    const AtomTokenSpan* span = atomSpanContainingCut(spans, cut);
    if (!span) {
        return cut;
    }
    ++adjustment_count;
    return span->begin;
}

size_t truncateEndBeforeSplitAtom(
    const std::vector<AtomTokenSpan>& spans,
    size_t window_begin,
    size_t desired_end,
    size_t capacity,
    const std::string& source,
    size_t& adjustment_count) {
    const AtomTokenSpan* span = atomSpanContainingCut(spans, desired_end);
    if (!span) {
        return desired_end;
    }
    if (span->begin <= window_begin) {
        throw std::runtime_error(
            source + ": typed atom span [" + std::to_string(span->begin) + "," +
            std::to_string(span->end) + ") length=" +
            std::to_string(span->end - span->begin) +
            " exceeds available window capacity=" + std::to_string(capacity));
    }
    ++adjustment_count;
    return span->begin;
}

std::shared_ptr<const GRIM::Goal> offsetGoalSpans(
    const std::shared_ptr<const GRIM::Goal>& source,
    std::int32_t offset) {
    if (!source || offset == 0) {
        return source;
    }

    auto shifted = std::make_shared<GRIM::Goal>();
    if (source->target_state.has_value()) {
        GRIM::TargetState target_state;
        target_state.token_ids = source->target_state->token_ids;
        target_state.span = GRIM::GoalTokenSpan{
            source->target_state->span.begin + offset,
            source->target_state->span.end + offset};
        shifted->target_state = std::move(target_state);
    }
    if (source->success_criteria.has_value()) {
        GRIM::SuccessCriteria criteria = *source->success_criteria;
        criteria.span.begin += offset;
        criteria.span.end += offset;
        for (auto& entry : criteria.entries) {
            entry.criterion_span.begin += offset;
            entry.criterion_span.end += offset;
            if (entry.evidence_span.valid()) {
                entry.evidence_span.begin += offset;
                entry.evidence_span.end += offset;
            }
        }
        shifted->success_criteria = std::move(criteria);
    }
    if (source->constraints.has_value()) {
        GRIM::Constraints constraints = *source->constraints;
        constraints.span.begin += offset;
        constraints.span.end += offset;
        for (auto& entry : constraints.entries) {
            entry.constraint_span.begin += offset;
            entry.constraint_span.end += offset;
        }
        shifted->constraints = std::move(constraints);
    }
    return shifted;
}

std::shared_ptr<const GRIM::ConceptBlockSpans> offsetConceptBlockSpans(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& source,
    std::int32_t offset) {
    if (!source || offset == 0) {
        return source;
    }
    auto shifted = std::make_shared<GRIM::ConceptBlockSpans>(*source);
    auto shift_entries = [offset](
        std::vector<GRIM::ConceptBlockSpanEntry>& entries) {
        for (auto& entry : entries) {
            entry.span.begin += offset;
            entry.span.end += offset;
        }
    };
    shift_entries(shifted->knowns);
    shift_entries(shifted->unknowns);
    std::shared_ptr<const GRIM::ConceptBlockSpans> immutable_spans =
        std::move(shifted);
    return immutable_spans;
}

bool conceptBlockSpansFitPrefix(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& spans,
    size_t source_end) {
    if (!spans) {
        return false;
    }
    auto entries_fit = [source_end](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries) {
        return std::all_of(
            entries.begin(), entries.end(),
            [source_end](const GRIM::ConceptBlockSpanEntry& entry) {
                return static_cast<size_t>(entry.span.end) <= source_end;
            });
    };
    return entries_fit(spans->knowns) && entries_fit(spans->unknowns);
}

std::shared_ptr<const GRIM::ConceptBlockSpans> sliceConceptBlockSpansForSftWindow(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& source,
    size_t prefix_length,
    size_t response_source_begin,
    size_t response_source_end) {
    if (!source) {
        return nullptr;
    }

    auto sliced = std::make_shared<GRIM::ConceptBlockSpans>();
    auto slice_entries = [prefix_length, response_source_begin, response_source_end](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries,
        std::vector<GRIM::ConceptBlockSpanEntry>& destination) {
        destination.reserve(entries.size());
        for (const auto& source_entry : entries) {
            GRIM::ConceptBlockSpanEntry entry = source_entry;
            if (entry.span.end <= static_cast<std::int32_t>(prefix_length)) {
                destination.push_back(std::move(entry));
                continue;
            }
            if (entry.span.begin >= static_cast<std::int32_t>(response_source_begin) &&
                entry.span.end <= static_cast<std::int32_t>(response_source_end)) {
                const std::int32_t offset = static_cast<std::int32_t>(prefix_length) -
                    static_cast<std::int32_t>(response_source_begin);
                entry.span.begin += offset;
                entry.span.end += offset;
                destination.push_back(std::move(entry));
            }
        }
    };
    slice_entries(source->knowns, sliced->knowns);
    slice_entries(source->unknowns, sliced->unknowns);
    if (sliced->empty()) {
        return nullptr;
    }
    std::shared_ptr<const GRIM::ConceptBlockSpans> immutable_spans =
        std::move(sliced);
    return immutable_spans;
}

bool goalFitsPrefix(const std::shared_ptr<const GRIM::Goal>& goal,
                    size_t source_end) {
    if (!goal) {
        return false;
    }
    if (goal->target_state.has_value() &&
        static_cast<size_t>(goal->target_state->span.end) > source_end) {
        return false;
    }
    if (goal->success_criteria.has_value() &&
        static_cast<size_t>(goal->success_criteria->span.end) > source_end) {
        return false;
    }
    if (goal->constraints.has_value() &&
        static_cast<size_t>(goal->constraints->span.end) > source_end) {
        return false;
    }
    return true;
}

void appendSftTokenRange(GrmtSequence& destination,
                         const GrmtSequence& source,
                         size_t begin,
                         size_t end) {
    destination.token_ids.insert(destination.token_ids.end(),
        source.token_ids.begin() + static_cast<ptrdiff_t>(begin),
        source.token_ids.begin() + static_cast<ptrdiff_t>(end));
    destination.targets.insert(destination.targets.end(),
        source.targets.begin() + static_cast<ptrdiff_t>(begin),
        source.targets.begin() + static_cast<ptrdiff_t>(end));
    destination.token_numeric_values.insert(destination.token_numeric_values.end(),
        source.token_numeric_values.begin() + static_cast<ptrdiff_t>(begin),
        source.token_numeric_values.begin() + static_cast<ptrdiff_t>(end));
    destination.token_atom_mask.insert(destination.token_atom_mask.end(),
        source.token_atom_mask.begin() + static_cast<ptrdiff_t>(begin),
        source.token_atom_mask.begin() + static_cast<ptrdiff_t>(end));
    destination.atom_entry_ids.insert(destination.atom_entry_ids.end(),
        source.atom_entry_ids.begin() + static_cast<ptrdiff_t>(begin),
        source.atom_entry_ids.begin() + static_cast<ptrdiff_t>(end));
    destination.token_local_atom_indices.insert(
        destination.token_local_atom_indices.end(),
        source.token_local_atom_indices.begin() + static_cast<ptrdiff_t>(begin),
        source.token_local_atom_indices.begin() + static_cast<ptrdiff_t>(end));
    destination.token_atom_flags.insert(destination.token_atom_flags.end(),
        source.token_atom_flags.begin() + static_cast<ptrdiff_t>(begin),
        source.token_atom_flags.begin() + static_cast<ptrdiff_t>(end));
    if (!source.token_exec_slot_indices.empty()) {
        destination.token_exec_slot_indices.insert(destination.token_exec_slot_indices.end(),
            source.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(begin),
            source.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(end));
    }
}

SftWindowConstruction constructSftWindows(
    const std::vector<GrmtSequence>& sequences,
    const std::string& split_name,
    int max_seq_len,
    int sliding_window_stride) {
    SftWindowConstruction result;
    result.sequences.reserve(sequences.size());

    const size_t response_overlap = static_cast<size_t>(
        max_seq_len - sliding_window_stride);

    for (const auto& sequence : sequences) {
        const size_t sequence_length = sequence.token_ids.size();
        const auto atom_spans = collectAtomTokenSpans(
            sequence.token_ids,
            "Sliding window (" + split_name + ", SFT)");
        if (sequence.prompt_length <= 0 || sequence.prompt_end_pos < 0) {
            throw std::runtime_error(
                "Sliding window (" + split_name +
                "): training_stage=sft requires a non-empty prompt span on every sequence");
        }
        if (sequence.prompt_end_pos >= static_cast<int32_t>(sequence_length) ||
            sequence.prompt_end_pos - sequence.prompt_length + 1 < 0) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): invalid prompt span");
        }

        // Pin the entire functional prompt through prompt_end_pos. For SFT,
        // this is every model-visible token before the answer. A configured
        // BOS may precede the authored span and remains part of this prefix.
        const size_t prefix_length =
            static_cast<size_t>(sequence.prompt_end_pos) + 1;
        if (prefix_length >= sequence_length) {
            throw std::runtime_error(
                "Sliding window (" + split_name +
                "): training_stage=sft prompt leaves no response tokens");
        }
        if (prefix_length >= static_cast<size_t>(max_seq_len)) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): SFT prefix length=" +
                std::to_string(prefix_length) +
                " leaves no response capacity within max_seq_len=" +
                std::to_string(max_seq_len));
        }
        for (size_t target_position = 1;
             target_position < prefix_length;
             ++target_position) {
            const size_t causal_row = target_position - 1;
            if (sequence.targets[causal_row] != -1) {
                throw std::runtime_error(
                    "Sliding window (" + split_name +
                    ", SFT): functional prompt token at position=" +
                    std::to_string(target_position) +
                    " has an LM target; only answer tokens may be supervised");
            }
        }

        const size_t response_length = sequence_length - prefix_length;
        const size_t response_capacity =
            static_cast<size_t>(max_seq_len) - prefix_length;
        const bool fragmented = response_length > response_capacity;
        if (atomSpanContainingCut(atom_spans, prefix_length)) {
            throw std::runtime_error(
                "Sliding window (" + split_name +
                ", SFT): pinned prompt prefix ends inside a typed atom span at token cut=" +
                std::to_string(prefix_length));
        }
        if (fragmented && response_overlap >= response_capacity) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): SFT response capacity=" +
                std::to_string(response_capacity) +
                " cannot preserve configured response overlap=" +
                std::to_string(response_overlap));
        }
        if (fragmented &&
            (sequence.execution_active ||
             sequence.execution_gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED)) {
            throw std::runtime_error(
                "Execution-control-supervised SFT sequence exceeds max_seq_len; "
                "execution-control rows cannot be fragmented");
        }

        const size_t response_hop = fragmented
            ? response_capacity - response_overlap
            : response_capacity;
        size_t nominal_response_begin = 0;
        size_t previous_response_end = 0;
        size_t previous_window_begin = std::numeric_limits<size_t>::max();
        bool first_window = true;
        if (fragmented) {
            ++result.long_sequence_count;
        }

        while (first_window || previous_response_end < response_length) {
            const size_t requested_response_begin = first_window
                ? nominal_response_begin
                : std::min(nominal_response_begin, previous_response_end);
            const size_t nominal_source_begin = prefix_length + requested_response_begin;
            size_t source_begin = rewindCutToAtomOpening(
                atom_spans,
                nominal_source_begin,
                result.atom_safe_cut_adjustments);
            size_t response_begin = source_begin - prefix_length;
            if (!first_window && response_begin == previous_window_begin) {
                const AtomTokenSpan* repeated_span =
                    atomSpanContainingCut(atom_spans, nominal_source_begin);
                if (!repeated_span) {
                    nominal_response_begin += response_hop;
                    continue;
                }
                source_begin = repeated_span->end;
                response_begin = source_begin - prefix_length;
            }

            const size_t desired_source_end =
                std::min(sequence_length, source_begin + response_capacity);
            const size_t source_end = truncateEndBeforeSplitAtom(
                atom_spans,
                source_begin,
                desired_source_end,
                response_capacity,
                "Sliding window (" + split_name + ", SFT)",
                result.atom_safe_cut_adjustments);
            if (source_end <= source_begin) {
                throw std::runtime_error(
                    "Sliding window (" + split_name +
                    ", SFT): atom-safe response window made no progress at source token index=" +
                    std::to_string(source_begin));
            }
            const size_t response_end = source_end - prefix_length;
            const size_t owned_response_begin = first_window
                ? 0
                : previous_response_end;
            if (owned_response_begin < response_begin) {
                throw std::runtime_error(
                    "Sliding window (" + split_name +
                    ", SFT): atom-safe response windows left a source-token gap before index=" +
                    std::to_string(source_begin));
            }

            GrmtSequence window;
            if (!fragmented) {
                window = sequence;
            } else {
                window.concept_block_id = sequence.concept_block_id;
                window.atom_table = sequence.atom_table;
                window.goal = sequence.goal;
                window.concept_block_spans = sliceConceptBlockSpansForSftWindow(
                    sequence.concept_block_spans,
                    prefix_length,
                    prefix_length + response_begin,
                    prefix_length + response_end);
                appendSftTokenRange(window, sequence, 0, prefix_length);
                appendSftTokenRange(
                    window,
                    sequence,
                    prefix_length + response_begin,
                    prefix_length + response_end);
                window.prompt_length = sequence.prompt_length;
                window.prompt_end_pos = sequence.prompt_end_pos;
            }

            // Visible prompt/overlap context is not supervision. Re-author the
            // targets so each newly owned response token is predicted once.
            std::fill(window.targets.begin(), window.targets.end(), -1);
            for (size_t response_token = owned_response_begin;
                 response_token < response_end;
                 ++response_token) {
                const size_t source_target_position =
                    prefix_length + response_token - 1;
                const size_t local_token_position =
                    prefix_length + response_token - response_begin;
                window.targets[local_token_position - 1] =
                    sequence.targets[source_target_position];
            }
            window.targets.back() = -1;

            const auto output_atom_spans = collectAtomTokenSpans(
                window.token_ids,
                "Sliding window (" + split_name + ", SFT output)");
            rebaseSequenceLocalAtoms(
                window,
                sequence.local_atom_table,
                "Sliding window (" + split_name + ", SFT output)");
            authorAtomAuxTargetMask(
                window,
                output_atom_spans,
                "Sliding window (" + split_name + ", SFT output)");

            result.sequences.push_back(std::move(window));
            if (fragmented) {
                ++result.generated_window_count;
            }

            previous_response_end = response_end;
            previous_window_begin = response_begin;
            if (response_end == response_length) {
                break;
            }
            nominal_response_begin += response_hop;
            first_window = false;
        }
    }

    return result;
}

} // namespace

void injectBoundaryTokens(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          GRIM::HyperParameters::TrainingStage training_stage,
                          bool add_bos_token,
                          bool add_eos_token,
                          size_t& added_bos_out,
                          size_t& added_eos_out) {
    added_bos_out = 0;
    added_eos_out = 0;

    for (auto& seq : sequences) {
        if (seq.token_ids.empty()) continue;

        // Add BOS if missing at start (controlled by config flag add_bos_token)
        if (add_bos_token && seq.token_ids.front() != GRIM::Tokenizer::BOS_TOKEN_ID) {
            const int old_first_token = seq.token_ids.front();
            seq.token_ids.insert(seq.token_ids.begin(), GRIM::Tokenizer::BOS_TOKEN_ID);
            seq.token_numeric_values.insert(seq.token_numeric_values.begin(), 0.0f);
            seq.token_atom_mask.insert(seq.token_atom_mask.begin(), 0);
            seq.atom_entry_ids.insert(seq.atom_entry_ids.begin(), GRIM::Tokenizer::kAtomEntryNone);
            seq.token_local_atom_indices.insert(
                seq.token_local_atom_indices.begin(),
                GRIM::Tokenizer::kLocalAtomIndexNone);
            seq.token_atom_flags.insert(seq.token_atom_flags.begin(), 0);
            const int bos_target =
                training_stage == GRIM::HyperParameters::TrainingStage::SFT
                    ? -1
                    : old_first_token;
            seq.targets.insert(seq.targets.begin(), bos_target);
            if (!seq.token_exec_slot_indices.empty())
                seq.token_exec_slot_indices.insert(seq.token_exec_slot_indices.begin(), static_cast<int32_t>(-1));
            // BOS insertion shifted all existing token positions right by 1.
            // Remap compiled_bootstrap_bindings token_pos to match.
            for (auto& b : seq.compiled_bootstrap_bindings)
                b.token_pos += 1;
            if (seq.prompt_end_pos >= 0) {
                seq.prompt_end_pos += 1;
            }
            seq.goal = offsetGoalSpans(seq.goal, 1);
            seq.concept_block_spans =
                offsetConceptBlockSpans(seq.concept_block_spans, 1);
            added_bos_out++;
        }

        // Add EOS if missing at end (controlled by config flag add_eos_token)
        if (add_eos_token && seq.token_ids.back() != GRIM::Tokenizer::EOS_TOKEN_ID) {
            seq.token_ids.push_back(GRIM::Tokenizer::EOS_TOKEN_ID);
            seq.token_numeric_values.push_back(0.0f);
            seq.token_atom_mask.push_back(0);
            seq.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
            seq.token_local_atom_indices.push_back(GRIM::Tokenizer::kLocalAtomIndexNone);
            seq.token_atom_flags.push_back(0);
            // Fix shift: the PREVIOUS position's target was -1 (no next token existed
            // when DataLoader ran). Now EOS follows it, so set target = EOS.
            if (!seq.targets.empty()) {
                seq.targets.back() = GRIM::Tokenizer::EOS_TOKEN_ID;  // position before EOS → predict EOS
            }
            seq.targets.push_back(-1);  // EOS position itself: nothing follows
            if (!seq.token_exec_slot_indices.empty())
                seq.token_exec_slot_indices.push_back(static_cast<int32_t>(-1));
            added_eos_out++;
        }
    }
}

void applySlidingWindows(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                         const std::string& split_name,
                         GRIM::HyperParameters::TrainingStage training_stage,
                         int max_seq_len,
                         int sliding_window_stride,
                         int min_seq_valid_tokens,
                         bool add_bos_token,
                         bool add_eos_token,
                         TrainingLogger& logger) {
    // Bracket sequences with BOS/EOS before windowing so window math sees
    // fully-bracketed input. Per-split summary is emitted here so each
    // train/val pass reports its own boundary-injection count.
    size_t added_bos = 0;
    size_t added_eos = 0;
    injectBoundaryTokens(
        sequences,
        training_stage,
        add_bos_token,
        add_eos_token,
        added_bos,
        added_eos);
    if (added_bos > 0 || added_eos > 0) {
        logger.log("[Data] Boundary tokens (" + split_name + "): added_bos=" +
                   std::to_string(added_bos) + " added_eos=" + std::to_string(added_eos));
    }

    if (training_stage == GRIM::HyperParameters::TrainingStage::SFT) {
        auto construction = constructSftWindows(
            sequences, split_name, max_seq_len, sliding_window_stride);
        sequences = std::move(construction.sequences);
        if (construction.long_sequence_count > 0) {
            logger.log("Sliding window (" + split_name + "): " +
                       std::to_string(construction.long_sequence_count) +
                       " long SFT sequences expanded into " +
                       std::to_string(construction.generated_window_count) +
                       " prompt-pinned response windows" +
                       " (atom_safe_cut_adjustments=" +
                       std::to_string(construction.atom_safe_cut_adjustments) + ")");
        }
        filterOverlongSequences(sequences, split_name, max_seq_len, logger);
        filterShortSequences(sequences, split_name, min_seq_valid_tokens, logger);
        return;
    }
    if (training_stage != GRIM::HyperParameters::TrainingStage::PT) {
        throw std::runtime_error(
            "Sliding window (" + split_name + "): training_stage=" +
            std::string(GRIM::HyperParameters::trainingStageToJsonString(training_stage)) +
            " has no authored window-construction policy");
    }

    // PT owns the full document stream. Prompt metadata, if present in a
    // reused artifact, must not change PT windowing or downstream pooling.
    for (auto& sequence : sequences) {
        sequence.prompt_length = 0;
        sequence.prompt_end_pos = -1;
    }

    std::vector<GRIM::TokenizerArtifacts::GrmtSequence> windowed;
    windowed.reserve(sequences.size());

    size_t long_seq_count = 0;
    size_t generated_windows = 0;
    size_t bos_prepended = 0;
    size_t atom_safe_cut_adjustments = 0;

    // Final-position autoregressive boundary mask. Required by
    // BatchPayload's Rule 20 invariant.
    auto MaskFinalTarget = [](GRIM::TokenizerArtifacts::GrmtSequence& seq) {
        if (!seq.targets.empty()) {
            seq.targets.back() = -1;
        }
    };

    for (const auto& seq : sequences) {
        const auto atom_spans = collectAtomTokenSpans(
            seq.token_ids,
            "Sliding window (" + split_name + ", PT)");
        if (static_cast<int>(seq.token_ids.size()) <= max_seq_len) {
            // Short sequence or exactly max_seq_len — no windowing.
            GRIM::TokenizerArtifacts::GrmtSequence copy = seq;
            MaskFinalTarget(copy);
            rebaseSequenceLocalAtoms(
                copy,
                seq.local_atom_table,
                "Sliding window (" + split_name + ", PT output)");
            authorAtomAuxTargetMask(
                copy,
                atom_spans,
                "Sliding window (" + split_name + ", PT output)");
            windowed.push_back(std::move(copy));
            continue;
        }

        // Execution-active rows MUST NOT be fragmented — compiled_bootstrap_bindings
        // and transition_targets are whole-sequence structures with no windowing semantics.
        if (seq.execution_active ||
            seq.execution_gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED) {
            throw std::runtime_error(
                "Execution-control-supervised sequence exceeds max_seq_len (" +
                std::to_string(seq.token_ids.size()) + " > " +
                std::to_string(max_seq_len) +
                "). Execution-control rows cannot be split by sliding window. "
                "Increase max_seq_len or shorten the source data.");
        }

        long_seq_count++;
        const size_t seq_len = seq.token_ids.size();
        const bool has_prompt_span = seq.prompt_length > 0;
        size_t prompt_start = 0;
        size_t prompt_end = 0;  // exclusive
        if (has_prompt_span) {
            if (seq.prompt_end_pos < 0 ||
                seq.prompt_end_pos >= static_cast<int32_t>(seq_len)) {
                throw std::runtime_error(
                    "Sliding window (" + split_name + "): invalid prompt_end_pos=" +
                    std::to_string(seq.prompt_end_pos) + " for sequence length=" +
                    std::to_string(seq_len));
            }
            const int32_t derived_start =
                seq.prompt_end_pos - seq.prompt_length + 1;
            if (derived_start < 0) {
                throw std::runtime_error(
                    "Sliding window (" + split_name + "): prompt span extends before sequence start");
            }
            prompt_start = static_cast<size_t>(derived_start);
            prompt_end = static_cast<size_t>(seq.prompt_end_pos) + 1;
        }
        bool prompt_span_assigned = !has_prompt_span;
        size_t nominal_start = 0;
        size_t previous_window_begin = std::numeric_limits<size_t>::max();
        const size_t stride = static_cast<size_t>(sliding_window_stride);
        bool is_first_window = true;
        size_t prev_source_end = 0;  // Track previous window's source end for overlap masking

        while (is_first_window || prev_source_end < seq_len) {
            const size_t requested_start = is_first_window
                ? nominal_start
                : std::min(nominal_start, prev_source_end);
            size_t start = rewindCutToAtomOpening(
                atom_spans,
                requested_start,
                atom_safe_cut_adjustments);
            if (!is_first_window && start == previous_window_begin) {
                const AtomTokenSpan* repeated_span =
                    atomSpanContainingCut(atom_spans, requested_start);
                if (!repeated_span) {
                    nominal_start += stride;
                    continue;
                }
                start = repeated_span->end;
            }

            // Reserve 1 token for BOS if this is not the first window and BOS is enabled
            const bool prepend_bos = !is_first_window && add_bos_token;
            const size_t effective_max = (is_first_window || !prepend_bos)
                ? static_cast<size_t>(max_seq_len)
                : static_cast<size_t>(max_seq_len - 1);
            const size_t desired_end = std::min(seq_len, start + effective_max);
            const size_t end = truncateEndBeforeSplitAtom(
                atom_spans,
                start,
                desired_end,
                effective_max,
                "Sliding window (" + split_name + ", PT)",
                atom_safe_cut_adjustments);
            if (end <= start) {
                throw std::runtime_error(
                    "Sliding window (" + split_name +
                    ", PT): atom-safe window made no progress at source token index=" +
                    std::to_string(start));
            }
            if (!is_first_window && start > prev_source_end) {
                throw std::runtime_error(
                    "Sliding window (" + split_name +
                    ", PT): atom-safe windows left a source-token gap before index=" +
                    std::to_string(start));
            }

            GRIM::TokenizerArtifacts::GrmtSequence window;
            window.concept_block_id = seq.concept_block_id;
            // PT windows do not pin goal metadata. Retain it only on the
            // leading window when every authored logical span is present.
            if (is_first_window && goalFitsPrefix(seq.goal, end)) {
                window.goal = seq.goal;
            }
            if (is_first_window &&
                conceptBlockSpansFitPrefix(seq.concept_block_spans, end)) {
                window.concept_block_spans = seq.concept_block_spans;
            }

            // For non-first windows, prepend BOS token (gated on add_bos_token config)
            if (prepend_bos) {
                window.token_ids.push_back(GRIM::Tokenizer::BOS_TOKEN_ID);
                // BOS predicts the first token of this local window.
                window.targets.push_back(seq.token_ids[start]);
                window.token_numeric_values.push_back(0.0f);
                window.token_atom_mask.push_back(0);
                window.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
                window.token_local_atom_indices.push_back(
                    GRIM::Tokenizer::kLocalAtomIndexNone);
                window.token_atom_flags.push_back(0);
                if (!seq.token_exec_slot_indices.empty())
                    window.token_exec_slot_indices.push_back(static_cast<int32_t>(-1));
                bos_prepended++;
            }

            // Copy window content
            window.token_ids.insert(window.token_ids.end(),
                seq.token_ids.begin() + start, seq.token_ids.begin() + end);
            window.targets.insert(window.targets.end(),
                seq.targets.begin() + start, seq.targets.begin() + end);
            window.token_numeric_values.insert(window.token_numeric_values.end(),
                seq.token_numeric_values.begin() + start, seq.token_numeric_values.begin() + end);
            window.token_atom_mask.insert(window.token_atom_mask.end(),
                seq.token_atom_mask.begin() + start, seq.token_atom_mask.begin() + end);
            // Atom side channel — share parent sequence's AtomTable
            window.atom_table = seq.atom_table;
            window.atom_entry_ids.insert(window.atom_entry_ids.end(),
                seq.atom_entry_ids.begin() + start, seq.atom_entry_ids.begin() + end);
            window.token_local_atom_indices.insert(
                window.token_local_atom_indices.end(),
                seq.token_local_atom_indices.begin() + static_cast<ptrdiff_t>(start),
                seq.token_local_atom_indices.begin() + static_cast<ptrdiff_t>(end));
            window.token_atom_flags.insert(window.token_atom_flags.end(),
                seq.token_atom_flags.begin() + start, seq.token_atom_flags.begin() + end);

            if (!seq.token_exec_slot_indices.empty()) {
                window.token_exec_slot_indices.insert(window.token_exec_slot_indices.end(),
                    seq.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(start),
                    seq.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(end));
            }
            // Issue #143: Mask overlap prefix targets in non-first windows.
            // With stride < max_seq_len, the first (prev_source_end - start)
            // tokens were already trained on in the previous window. Mask them
            // to prevent double-training on the same targets.
            //
            // Issue #147: Subtract 1 from overlap_len. The position at
            // (prev_source_end - 1) was the LAST position in the previous
            // window, which was already  masked there (last-position mask).
            // Its target was NEVER trained. If we mask it here too, we create
            // a one-token training gap at every window boundary. By reducing
            // overlap by 1, this window trains that target instead.
            if (!is_first_window && prev_source_end > start) {
                const size_t raw_overlap = prev_source_end - start;
                const size_t overlap_len = (raw_overlap > 0) ? (raw_overlap - 1) : 0;
                // If we prepended a synthetic BOS for this local window, keep its
                // BOS→first-token supervision and only mask true overlapped source
                // positions after it.
                const size_t bos_offset = prepend_bos ? 1 : 0;
                for (size_t i = bos_offset; i < bos_offset + overlap_len && i < window.targets.size(); ++i) {
                    window.targets[i] = -1;
                }
            }

            // Mask last position for window boundary.
            // BatchPayload requires targets.back() == -1 unconditionally;
            // val no longer gets a special-cased "keep the final target"
            // path because BatchPayload would have masked it anyway. Window
            // boundaries stay as pure masked truncations; do not synthesize
            // EOS tokens at non-final window ends.
            if (!window.targets.empty()) {
                window.targets.back() = -1;
            }

            // Variable-length window — BatchPayload owns padding.
            // Logical prompt delimiters are side-channel positions, not token
            // IDs. Preserve the complete prompt on the first window that owns
            // it; partial overlaps must not masquerade as a complete prompt.
            if (!prompt_span_assigned &&
                prompt_start >= start && prompt_end <= end) {
                const size_t local_offset = prepend_bos ? 1 : 0;
                const size_t local_prompt_start =
                    local_offset + (prompt_start - start);
                const size_t local_prompt_end =
                    local_offset + (prompt_end - start);
                window.prompt_length = static_cast<int32_t>(
                    local_prompt_end - local_prompt_start);
                window.prompt_end_pos = static_cast<int32_t>(local_prompt_end - 1);
                prompt_span_assigned = true;
            }

            const auto output_atom_spans = collectAtomTokenSpans(
                window.token_ids,
                "Sliding window (" + split_name + ", PT output)");
            rebaseSequenceLocalAtoms(
                window,
                seq.local_atom_table,
                "Sliding window (" + split_name + ", PT output)");
            authorAtomAuxTargetMask(
                window,
                output_atom_spans,
                "Sliding window (" + split_name + ", PT output)");

            windowed.push_back(std::move(window));
            generated_windows++;

            prev_source_end = end;  // Track for overlap masking in next window
            previous_window_begin = start;
            if (end == seq_len) break;
            nominal_start += stride;
            is_first_window = false;
        }

        if (!prompt_span_assigned) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): no window contains the complete prompt span; "
                "increase max_seq_len or shorten the prompt");
        }
    }

    sequences = std::move(windowed);
    if (long_seq_count > 0) {
        logger.log("Sliding window (" + split_name + "): " +
                   std::to_string(long_seq_count) + " long sequences expanded into " +
                   std::to_string(generated_windows) + " windows" +
                   " (BOS prepended to " + std::to_string(bos_prepended) +
                   " mid-sequence windows, atom_safe_cut_adjustments=" +
                   std::to_string(atom_safe_cut_adjustments) + ")");
    }

    // Post-window cleanup. Both filters exist because windowing + BatchPayload
    // masking can leave sequences that downstream code can't consume.
    filterOverlongSequences(sequences, split_name, max_seq_len, logger);
    filterShortSequences(sequences, split_name, min_seq_valid_tokens, logger);
}

void filterOverlongSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                             const std::string& split_name,
                             int max_seq_len,
                             TrainingLogger& logger) {
    // HARD FILTER: Remove any sequences still exceeding max_seq_len after sliding window.
    // This catches cached .grmt files with old sequence lengths.
    const size_t before = sequences.size();
    sequences.erase(
        std::remove_if(sequences.begin(), sequences.end(),
            [max_seq_len](const GRIM::TokenizerArtifacts::GrmtSequence& seq) {
                return static_cast<int>(seq.token_ids.size()) > max_seq_len;
            }),
        sequences.end());
    const size_t removed = before - sequences.size();
    if (removed > 0) {
        logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) +
                   " sequences exceeding max_seq_len=" + std::to_string(max_seq_len));
    }
}

void filterShortSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          const std::string& split_name,
                          int min_seq_valid_tokens,
                          TrainingLogger& logger) {
    // HARD FILTER: Remove sequences with too few valid tokens after masking.
    // Prevents "valid_tokens=0" errors during loss computation.
    // Count the targets that are ALREADY authored as valid (>= 0); do not hardcode
    // positional assumptions like "index 0 is always BOS" because BOS insertion is
    // config-driven and BOS→first-token is valid supervision.
    if (min_seq_valid_tokens <= 0) return;
    const size_t before = sequences.size();
    sequences.erase(
        std::remove_if(sequences.begin(), sequences.end(),
            [min_seq_valid_tokens](const GRIM::TokenizerArtifacts::GrmtSequence& seq) {
                int valid = 0;
                for (size_t i = 0; i < seq.targets.size(); ++i) {
                    if (seq.targets[i] >= 0) valid++;
                }
                return valid < min_seq_valid_tokens;
            }),
        sequences.end());
    const size_t removed = before - sequences.size();
    if (removed > 0) {
        logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) +
                   " sequences with < " + std::to_string(min_seq_valid_tokens) + " valid tokens");
    }
}

} // namespace GRIMText::Training
