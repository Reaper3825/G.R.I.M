//======================================================//
//  AtomInsertionDecode.cpp
//  Request-local OPEN-type + EXIT span state machine
//======================================================//

#include "AtomInsertionDecode.hpp"

#include "AtomInsertionDecisionLayout.hpp"
#include "../UnigramByte/AtomTable.hpp"

#include <cmath>
#include <stdexcept>
#include <string_view>

namespace GRIM::AtomInsertion {
namespace {

struct DecodedAtomSpan {
    std::size_t begin_gap = 0;
    std::size_t end_gap = 0;
    Tokenizer::AtomType type = Tokenizer::AtomType::ATOM_INT;
};

} // namespace

std::string decodeAtomDecisionPredictions(
    const std::string& plain_text,
    const std::vector<std::uint8_t>& valid_gap_mask,
    const std::vector<float>& decision_logits,
    float decision_logit) {
    const std::size_t gap_count = plain_text.size() + 1;
    if (valid_gap_mask.size() != gap_count) {
        throw std::runtime_error(
            "decodeAtomDecisionPredictions: valid-gap mask size mismatch");
    }
    if (decision_logits.size() !=
        gap_count * static_cast<std::size_t>(kAtomDecisionClassCount)) {
        throw std::runtime_error(
            "decodeAtomDecisionPredictions: decision-logit shape mismatch");
    }
    if (!std::isfinite(decision_logit)) {
        throw std::runtime_error(
            "decodeAtomDecisionPredictions: decision logit is not finite");
    }

    std::vector<DecodedAtomSpan> spans;
    bool has_active_span = false;
    Tokenizer::AtomType active_type = Tokenizer::AtomType::ATOM_INT;
    std::size_t active_begin_gap = 0;

    for (std::size_t gap = 0; gap < gap_count; ++gap) {
        const float* row = decision_logits.data() +
            gap * static_cast<std::size_t>(kAtomDecisionClassCount);
        for (int decision_class = 0;
             decision_class < kAtomDecisionClassCount;
             ++decision_class) {
            if (!std::isfinite(row[decision_class])) {
                throw std::runtime_error(
                    "decodeAtomDecisionPredictions: non-finite decision logit at gap=" +
                    std::to_string(gap) + " class=" +
                    std::to_string(decision_class));
            }
        }
        if (valid_gap_mask[gap] == 0) {
            continue;
        }

        // EXIT is consumed before OPEN so adjacent spans can close and open at
        // one gap. The consumed flag prevents that same EXIT from immediately
        // closing the newly opened span.
        bool consumed_exit_at_gap = false;
        if (has_active_span && row[kExitDecisionClassIndex] >= decision_logit) {
            const auto parsed = Tokenizer::AtomTable::parseAtom(
                active_type,
                std::string_view(
                    plain_text.data() + active_begin_gap,
                    gap - active_begin_gap));
            if (parsed.success) {
                spans.push_back(
                    DecodedAtomSpan{active_begin_gap, gap, active_type});
            }
            has_active_span = false;
            consumed_exit_at_gap = true;
        }

        if (!has_active_span) {
            int selected_type_index = -1;
            float selected_logit = decision_logit;
            for (int type_index = 0;
                 type_index < kOpenDecisionClassCount;
                 ++type_index) {
                const float open_logit = row[type_index];
                if (open_logit >= decision_logit &&
                    (selected_type_index < 0 || open_logit > selected_logit)) {
                    selected_type_index = type_index;
                    selected_logit = open_logit;
                }
            }
            if (selected_type_index >= 0) {
                const auto selected_type =
                    static_cast<Tokenizer::AtomType>(selected_type_index);
                const bool predicts_same_gap_exit =
                    !consumed_exit_at_gap &&
                    row[kExitDecisionClassIndex] >= decision_logit;
                if (predicts_same_gap_exit) {
                    const auto parsed = Tokenizer::AtomTable::parseAtom(
                        selected_type, std::string_view{});
                    if (parsed.success) {
                        spans.push_back(
                            DecodedAtomSpan{gap, gap, selected_type});
                    }
                } else {
                    has_active_span = true;
                    active_type = selected_type;
                    active_begin_gap = gap;
                }
            }
        }
    }

    std::vector<int> close_at(gap_count, -1);
    std::vector<int> empty_at(gap_count, -1);
    std::vector<int> open_at(gap_count, -1);
    for (const DecodedAtomSpan& span : spans) {
        const int type_index = static_cast<int>(span.type);
        if (span.begin_gap == span.end_gap) {
            if (empty_at[span.begin_gap] >= 0) {
                throw std::runtime_error(
                    "decodeAtomDecisionPredictions: duplicate empty span event");
            }
            empty_at[span.begin_gap] = type_index;
            continue;
        }
        if (open_at[span.begin_gap] >= 0 || close_at[span.end_gap] >= 0) {
            throw std::runtime_error(
                "decodeAtomDecisionPredictions: overlapping span events");
        }
        open_at[span.begin_gap] = type_index;
        close_at[span.end_gap] = type_index;
    }

    std::string annotated;
    annotated.reserve(plain_text.size() + spans.size() * 18);
    for (std::size_t gap = 0; gap < gap_count; ++gap) {
        if (close_at[gap] >= 0) {
            annotated += Tokenizer::atomTokenText(
                Tokenizer::atomTypeToCloseTokenId(
                    static_cast<Tokenizer::AtomType>(close_at[gap])));
        }
        if (empty_at[gap] >= 0) {
            const auto type =
                static_cast<Tokenizer::AtomType>(empty_at[gap]);
            annotated += Tokenizer::atomTokenText(
                Tokenizer::atomTypeToOpenTokenId(type));
            annotated += Tokenizer::atomTokenText(
                Tokenizer::atomTypeToCloseTokenId(type));
        }
        if (open_at[gap] >= 0) {
            annotated += Tokenizer::atomTokenText(
                Tokenizer::atomTypeToOpenTokenId(
                    static_cast<Tokenizer::AtomType>(open_at[gap])));
        }
        if (gap < plain_text.size()) {
            annotated.push_back(plain_text[gap]);
        }
    }
    return annotated;
}

} // namespace GRIM::AtomInsertion
