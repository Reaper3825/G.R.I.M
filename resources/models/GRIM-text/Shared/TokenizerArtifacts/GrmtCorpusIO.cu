#include "GrmtCorpusIO.hpp"

#include "../ConceptBlock/ConceptBlockSpans.hpp"
#include "../Goal/Goal.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace GRIM::TokenizerArtifacts {
namespace {

std::string pathString(const fs::path& path) {
    const std::string s = path.string();
    if (s.empty()) {
        throw std::runtime_error("[GRMT] path is empty");
    }
    return s;
}

void ensureParentDirectory(const fs::path& path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return;
    }
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to create parent directory for " +
                                 path.string() + ": " + ec.message());
    }
}

void writeExact(std::ostream& output, const void* data, std::size_t bytes, const std::string& sink) {
    output.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!output) {
        throw std::runtime_error("[GRMT] failed write to " + sink + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

template <typename T>
void writeScalar(std::ostream& output, const T& value, const std::string& sink) {
    writeExact(output, &value, sizeof(T), sink);
}

void readExact(std::istream& input, void* data, std::size_t bytes, const std::string& source) {
    input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("[GRMT] failed read from " + source + " (bytes=" +
                                 std::to_string(bytes) + ")");
    }
}

template <typename T>
T readScalar(std::istream& input, const std::string& source) {
    T value{};
    readExact(input, &value, sizeof(T), source);
    return value;
}

void writeSlotId(
    std::ostream& output,
    GRIM::Execution::SlotId id,
    const std::string& sink)
{
    writeScalar(output, id.serialized(), sink);
}

GRIM::Execution::SlotId readSlotId(
    std::istream& input,
    const std::string& source)
{
    return GRIM::Execution::SlotId::fromSerialized(
        readScalar<GRIM::Execution::SlotId::Storage>(input, source));
}

void writeSlotIndex(
    std::ostream& output,
    GRIM::Execution::SlotIndex index,
    const std::string& sink)
{
    writeScalar(output, index.dense(), sink);
}

GRIM::Execution::SlotIndex readSlotIndex(
    std::istream& input,
    const std::string& source)
{
    return GRIM::Execution::SlotIndex::fromDense(
        readScalar<GRIM::Execution::SlotIndex::Storage>(input, source));
}

void writeTransitionId(
    std::ostream& output,
    GRIM::Execution::TransitionId id,
    const std::string& sink)
{
    writeScalar(output, id.serialized(), sink);
}

GRIM::Execution::TransitionId readTransitionId(
    std::istream& input,
    const std::string& source)
{
    return GRIM::Execution::TransitionId::fromSerialized(
        readScalar<GRIM::Execution::TransitionId::Storage>(input, source));
}

void writeTransitionIndex(
    std::ostream& output,
    GRIM::Execution::TransitionIndex index,
    const std::string& sink)
{
    writeScalar(output, index.dense(), sink);
}

GRIM::Execution::TransitionIndex readTransitionIndex(
    std::istream& input,
    const std::string& source)
{
    return GRIM::Execution::TransitionIndex::fromDense(
        readScalar<GRIM::Execution::TransitionIndex::Storage>(input, source));
}

std::uint32_t checkedCount(std::size_t count,
                           const std::string& field,
                           const std::string& sink) {
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(
            "[GRMT] " + field + " is too large to serialize in " + sink);
    }
    return static_cast<std::uint32_t>(count);
}

void validateGoalMetadata(const std::shared_ptr<const GRIM::Goal>& goal,
                          const std::string& source) {
    if (!goal) {
        return;
    }
    if (!goal->target_state.has_value() &&
        !goal->success_criteria.has_value() &&
        !goal->constraints.has_value()) {
        throw std::runtime_error(
            "[GRMT] " + source +
            ": goal has no target_state, success_criteria, or constraints");
    }

    auto validate_token_ids = [&source](const std::vector<std::int32_t>& token_ids,
                                        const GRIM::GoalTokenSpan& span,
                                        const std::string& field) {
        if (token_ids.empty()) {
            throw std::runtime_error(
                "[GRMT] " + source + ": " + field + " has no tokens");
        }
        if (!span.valid() ||
            static_cast<std::size_t>(span.length()) != token_ids.size()) {
            throw std::runtime_error(
                "[GRMT] " + source + ": " + field +
                " logical span does not match its token count");
        }
        for (std::size_t index = 0; index < token_ids.size(); ++index) {
            if (token_ids[index] < 0) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + field + " contains negative token id at index=" +
                    std::to_string(index));
            }
        }
    };
    auto validate_optional_token_ids =
        [&source, &validate_token_ids](
            const std::vector<std::int32_t>& token_ids,
            const GRIM::GoalTokenSpan& span,
            const std::string& field) {
            if (token_ids.empty()) {
                if (span.begin != -1 || span.end != -1) {
                    throw std::runtime_error(
                        "[GRMT] " + source + ": " + field +
                        " absent tokens require an absent logical span");
                }
                return false;
            }
            validate_token_ids(token_ids, span, field);
            return true;
        };

    if (goal->target_state.has_value()) {
        validate_token_ids(
            goal->target_state->token_ids,
            goal->target_state->span,
            "goal.target_state");
    }
    if (goal->success_criteria.has_value()) {
        const auto& criteria = *goal->success_criteria;
        const auto& entries = criteria.entries;
        if (entries.empty()) {
            throw std::runtime_error(
                "[GRMT] " + source + ": goal.success_criteria has no entries");
        }
        if (!criteria.span.valid()) {
            throw std::runtime_error(
                "[GRMT] " + source + ": goal.success_criteria span is invalid");
        }
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const std::string prefix =
                "goal.success_criteria[" + std::to_string(index) + "]";
            validate_token_ids(
                entries[index].token_ids,
                entries[index].criterion_span,
                prefix + ".criterion");
            const bool has_evidence = validate_optional_token_ids(
                entries[index].evidence_token_ids,
                entries[index].evidence_span,
                prefix + ".evidence");
            if (has_evidence && entries[index].criterion_span.end >
                entries[index].evidence_span.begin) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + prefix +
                    " criterion/evidence spans are out of order");
            }
            if (index > 0) {
                const auto& previous = entries[index - 1];
                const std::int32_t previous_end = previous.evidence_span.valid()
                    ? previous.evidence_span.end
                    : previous.criterion_span.end;
                if (previous_end > entries[index].criterion_span.begin) {
                    throw std::runtime_error(
                        "[GRMT] " + source +
                        ": success-criterion pairs overlap or are out of order");
                }
            }
        }
        const auto& last_entry = entries.back();
        const std::int32_t last_end = last_entry.evidence_span.valid()
            ? last_entry.evidence_span.end
            : last_entry.criterion_span.end;
        if (criteria.span.begin != entries.front().criterion_span.begin ||
            criteria.span.end != last_end) {
            throw std::runtime_error(
                "[GRMT] " + source +
                ": outer criteria span does not enclose the ordered pairs exactly");
        }
    }
    if (goal->constraints.has_value()) {
        const auto& entries = goal->constraints->entries;
        if (entries.empty()) {
            throw std::runtime_error(
                "[GRMT] " + source + ": goal.constraints has no entries");
        }
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const std::string field =
                "goal.constraints[" + std::to_string(index) + "]";
            validate_token_ids(
                entries[index].token_ids,
                entries[index].constraint_span,
                field);
            if (index > 0 &&
                entries[index - 1].constraint_span.end >
                    entries[index].constraint_span.begin) {
                throw std::runtime_error(
                    "[GRMT] " + source +
                    ": constraint spans overlap or are out of order");
            }
        }
    }
}

void validateGoalTokenSlices(const std::shared_ptr<const GRIM::Goal>& goal,
                             const std::vector<int>& sequence_token_ids,
                             const std::string& source) {
    if (!goal) {
        return;
    }
    auto validate = [&sequence_token_ids, &source](
        const std::vector<std::int32_t>& expected,
        const GRIM::GoalTokenSpan& span,
        const std::string& field) {
        if (static_cast<std::size_t>(span.end) > sequence_token_ids.size() ||
            !std::equal(
                expected.begin(), expected.end(),
                sequence_token_ids.begin() + span.begin)) {
            throw std::runtime_error(
                "[GRMT] " + source + ": " + field +
                " tokens do not match the delimited sequence span");
        }
    };
    if (goal->target_state.has_value()) {
        validate(goal->target_state->token_ids,
                 goal->target_state->span,
                 "goal.target_state");
    }
    if (goal->success_criteria.has_value()) {
        for (std::size_t index = 0;
             index < goal->success_criteria->entries.size();
             ++index) {
            const auto& entry = goal->success_criteria->entries[index];
            const std::string prefix =
                "goal.success_criteria[" + std::to_string(index) + "]";
            validate(entry.token_ids, entry.criterion_span,
                     prefix + ".criterion");
            if (!entry.evidence_token_ids.empty()) {
                validate(entry.evidence_token_ids, entry.evidence_span,
                         prefix + ".evidence");
            }
        }
    }
    if (goal->constraints.has_value()) {
        for (std::size_t index = 0;
             index < goal->constraints->entries.size();
             ++index) {
            const auto& entry = goal->constraints->entries[index];
            validate(
                entry.token_ids,
                entry.constraint_span,
                "goal.constraints[" + std::to_string(index) + "]");
        }
    }
}

void validateGoalTokenRange(const std::shared_ptr<const GRIM::Goal>& goal,
                            std::uint32_t vocab_size,
                            const std::string& source) {
    if (!goal) {
        return;
    }
    auto validate = [vocab_size, &source](
        const std::vector<std::int32_t>& token_ids,
        const std::string& field) {
        for (std::size_t index = 0; index < token_ids.size(); ++index) {
            if (static_cast<std::uint32_t>(token_ids[index]) >= vocab_size) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + field + " token id=" +
                    std::to_string(token_ids[index]) + " at index=" +
                    std::to_string(index) + " is outside vocab_size=" +
                    std::to_string(vocab_size));
            }
        }
    };

    if (goal->target_state.has_value()) {
        validate(goal->target_state->token_ids, "goal.target_state");
    }
    if (goal->success_criteria.has_value()) {
        for (std::size_t index = 0;
             index < goal->success_criteria->entries.size();
             ++index) {
            const auto& entry = goal->success_criteria->entries[index];
            const std::string prefix =
                "goal.success_criteria[" + std::to_string(index) + "]";
            validate(entry.token_ids, prefix + ".criterion");
            validate(entry.evidence_token_ids, prefix + ".evidence");
        }
    }
    if (goal->constraints.has_value()) {
        for (std::size_t index = 0;
             index < goal->constraints->entries.size();
             ++index) {
            validate(
                goal->constraints->entries[index].token_ids,
                "goal.constraints[" + std::to_string(index) + "]");
        }
    }
}

void validateConceptBlockSpanMetadata(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& spans,
    const std::string& source) {
    if (!spans) {
        return;
    }
    if (spans->empty()) {
        throw std::runtime_error(
            "[GRMT] " + source +
            ": concept_block_spans has no known or unknown entries");
    }

    auto validate_entries = [&source](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries,
        const char* field) {
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            const std::string entry_field =
                std::string(field) + "[" + std::to_string(index) + "]";
            if (entry.token_ids.empty()) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + entry_field +
                    " has no tokens");
            }
            if (!entry.span.valid() ||
                static_cast<std::size_t>(entry.span.length()) !=
                    entry.token_ids.size()) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + entry_field +
                    " logical span does not match its token count");
            }
            for (std::size_t token = 0; token < entry.token_ids.size(); ++token) {
                if (entry.token_ids[token] < 0) {
                    throw std::runtime_error(
                        "[GRMT] " + source + ": " + entry_field +
                        " contains negative token id at index=" +
                        std::to_string(token));
                }
            }
            if (index > 0 && entries[index - 1].span.end > entry.span.begin) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + field +
                    " spans overlap or are out of order");
            }
        }
    };

    validate_entries(spans->knowns, "knowns");
    validate_entries(spans->unknowns, "unknowns");
    if (!spans->knowns.empty() && !spans->unknowns.empty() &&
        spans->knowns.back().span.end > spans->unknowns.front().span.begin) {
        throw std::runtime_error(
            "[GRMT] " + source +
            ": known spans must precede unknown spans");
    }
}

void validateConceptBlockSpanTokenSlices(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& spans,
    const std::vector<int>& sequence_token_ids,
    const std::string& source) {
    if (!spans) {
        return;
    }
    auto validate_entries = [&sequence_token_ids, &source](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries,
        const char* field) {
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (static_cast<std::size_t>(entry.span.end) >
                    sequence_token_ids.size() ||
                !std::equal(
                    entry.token_ids.begin(), entry.token_ids.end(),
                    sequence_token_ids.begin() + entry.span.begin)) {
                throw std::runtime_error(
                    "[GRMT] " + source + ": " + field + "[" +
                    std::to_string(index) +
                    "] tokens do not match the delimited sequence span");
            }
        }
    };
    validate_entries(spans->knowns, "knowns");
    validate_entries(spans->unknowns, "unknowns");
}

void validateConceptBlockSpanTokenRange(
    const std::shared_ptr<const GRIM::ConceptBlockSpans>& spans,
    std::uint32_t vocab_size,
    const std::string& source) {
    if (!spans) {
        return;
    }
    auto validate_entries = [vocab_size, &source](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries,
        const char* field) {
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& token_ids = entries[index].token_ids;
            for (std::size_t token = 0; token < token_ids.size(); ++token) {
                if (static_cast<std::uint32_t>(token_ids[token]) >= vocab_size) {
                    throw std::runtime_error(
                        "[GRMT] " + source + ": " + field + "[" +
                        std::to_string(index) + "] token id=" +
                        std::to_string(token_ids[token]) + " at index=" +
                        std::to_string(token) + " is outside vocab_size=" +
                        std::to_string(vocab_size));
                }
            }
        }
    };
    validate_entries(spans->knowns, "knowns");
    validate_entries(spans->unknowns, "unknowns");
}

void validateSequenceTokenRange(const GrmtSequence& sequence,
                                std::uint32_t vocab_size,
                                const std::string& source) {
    for (std::size_t index = 0; index < sequence.token_ids.size(); ++index) {
        const int token_id = sequence.token_ids[index];
        if (token_id < 0 || static_cast<std::uint32_t>(token_id) >= vocab_size) {
            throw std::runtime_error(
                "[GRMT] " + source + ": token_ids token id=" +
                std::to_string(token_id) + " at index=" +
                std::to_string(index) + " is outside vocab_size=" +
                std::to_string(vocab_size));
        }
    }

    for (std::size_t index = 0; index < sequence.targets.size(); ++index) {
        const int target = sequence.targets[index];
        if (target == -1) {
            continue;
        }
        if (target < 0 || static_cast<std::uint32_t>(target) >= vocab_size) {
            throw std::runtime_error(
                "[GRMT] " + source + ": targets token id=" +
                std::to_string(target) + " at index=" +
                std::to_string(index) + " is outside vocab_size=" +
                std::to_string(vocab_size) + " (only -1 is a valid sentinel)");
        }
    }
}

void writeTokenIds(std::ostream& output,
                   const std::vector<std::int32_t>& token_ids,
                   const std::string& field,
                   const std::string& sink) {
    const std::uint32_t count = checkedCount(token_ids.size(), field, sink);
    writeScalar(output, count, sink);
    writeExact(output, token_ids.data(),
               static_cast<std::size_t>(count) * sizeof(std::int32_t), sink);
}

std::vector<std::int32_t> readTokenIds(std::istream& input,
                                       const std::string& source) {
    const std::uint32_t count = readScalar<std::uint32_t>(input, source);
    std::vector<std::int32_t> token_ids(count);
    readExact(input, token_ids.data(),
              static_cast<std::size_t>(count) * sizeof(std::int32_t), source);
    return token_ids;
}

void writeGoalSpan(std::ostream& output,
                   const GRIM::GoalTokenSpan& span,
                   const std::string& sink) {
    writeScalar(output, span.begin, sink);
    writeScalar(output, span.end, sink);
}

GRIM::GoalTokenSpan readGoalSpan(std::istream& input,
                                 const std::string& source) {
    return GRIM::GoalTokenSpan{
        readScalar<std::int32_t>(input, source),
        readScalar<std::int32_t>(input, source)};
}

void writeGoalForSequence(std::ostream& output,
                          const GrmtSequence& sequence,
                          const std::string& sink) {
    validateGoalMetadata(sequence.goal, sink);
    const std::uint8_t has_goal = sequence.goal ? 1 : 0;
    writeScalar(output, has_goal, sink);
    if (!sequence.goal) {
        return;
    }

    const std::uint8_t has_target_state =
        sequence.goal->target_state.has_value() ? 1 : 0;
    writeScalar(output, has_target_state, sink);
    if (has_target_state != 0) {
        writeTokenIds(output, sequence.goal->target_state->token_ids,
                      "goal.target_state", sink);
        writeGoalSpan(output, sequence.goal->target_state->span, sink);
    }

    const std::uint8_t has_success_criteria =
        sequence.goal->success_criteria.has_value() ? 1 : 0;
    writeScalar(output, has_success_criteria, sink);
    if (has_success_criteria != 0) {
        const auto& entries = sequence.goal->success_criteria->entries;
        const std::uint32_t entry_count =
            checkedCount(entries.size(), "goal.success_criteria", sink);
        writeGoalSpan(output, sequence.goal->success_criteria->span, sink);
        writeScalar(output, entry_count, sink);
        for (const auto& entry : entries) {
            writeTokenIds(output, entry.token_ids,
                          "goal.success_criteria.criterion", sink);
            writeGoalSpan(output, entry.criterion_span, sink);
            writeTokenIds(output, entry.evidence_token_ids,
                          "goal.success_criteria.evidence", sink);
            writeGoalSpan(output, entry.evidence_span, sink);
        }
    }

    const std::uint8_t has_constraints =
        sequence.goal->constraints.has_value() ? 1 : 0;
    writeScalar(output, has_constraints, sink);
    if (has_constraints != 0) {
        const auto& entries = sequence.goal->constraints->entries;
        const std::uint32_t entry_count =
            checkedCount(entries.size(), "goal.constraints", sink);
        writeScalar(output, entry_count, sink);
        for (const auto& entry : entries) {
            writeTokenIds(output, entry.token_ids,
                          "goal.constraints", sink);
            writeGoalSpan(output, entry.constraint_span, sink);
        }
    }
}

std::shared_ptr<const GRIM::Goal> readGoalForSequence(
    std::istream& input,
    const std::string& source) {
    const std::uint8_t has_goal = readScalar<std::uint8_t>(input, source);
    if (has_goal > 1) {
        throw std::runtime_error(
            "[GRMT] invalid goal flag in " + source);
    }
    if (has_goal == 0) {
        return nullptr;
    }

    auto goal = std::make_shared<GRIM::Goal>();
    const std::uint8_t has_target_state =
        readScalar<std::uint8_t>(input, source);
    if (has_target_state > 1) {
        throw std::runtime_error(
            "[GRMT] invalid target_state flag in " + source);
    }
    if (has_target_state != 0) {
        GRIM::TargetState target_state;
        target_state.token_ids = readTokenIds(input, source);
        target_state.span = readGoalSpan(input, source);
        goal->target_state = std::move(target_state);
    }

    const std::uint8_t has_success_criteria =
        readScalar<std::uint8_t>(input, source);
    if (has_success_criteria > 1) {
        throw std::runtime_error(
            "[GRMT] invalid success_criteria flag in " + source);
    }
    if (has_success_criteria != 0) {
        GRIM::SuccessCriteria success_criteria;
        success_criteria.span = readGoalSpan(input, source);
        const std::uint32_t entry_count =
            readScalar<std::uint32_t>(input, source);
        success_criteria.entries.reserve(entry_count);
        for (std::uint32_t index = 0; index < entry_count; ++index) {
            GRIM::SuccessCriterion entry;
            entry.token_ids = readTokenIds(input, source);
            entry.criterion_span = readGoalSpan(input, source);
            entry.evidence_token_ids = readTokenIds(input, source);
            entry.evidence_span = readGoalSpan(input, source);
            success_criteria.entries.push_back(std::move(entry));
        }
        goal->success_criteria = std::move(success_criteria);
    }

    const std::uint8_t has_constraints =
        readScalar<std::uint8_t>(input, source);
    if (has_constraints > 1) {
        throw std::runtime_error(
            "[GRMT] invalid constraints flag in " + source);
    }
    if (has_constraints != 0) {
        GRIM::Constraints constraints;
        const std::uint32_t entry_count =
            readScalar<std::uint32_t>(input, source);
        constraints.entries.reserve(entry_count);
        for (std::uint32_t index = 0; index < entry_count; ++index) {
            GRIM::Constraint entry;
            entry.token_ids = readTokenIds(input, source);
            entry.constraint_span = readGoalSpan(input, source);
            constraints.entries.push_back(std::move(entry));
        }
        goal->constraints = std::move(constraints);
    }

    std::shared_ptr<const GRIM::Goal> immutable_goal = std::move(goal);
    validateGoalMetadata(immutable_goal, source);
    return immutable_goal;
}

void writeConceptBlockSpansForSequence(
    std::ostream& output,
    const GrmtSequence& sequence,
    const std::string& sink) {
    validateConceptBlockSpanMetadata(sequence.concept_block_spans, sink);
    auto write_entries = [&output, &sink](
        const std::vector<GRIM::ConceptBlockSpanEntry>& entries,
        const char* field) {
        const std::uint32_t entry_count =
            checkedCount(entries.size(), field, sink);
        writeScalar(output, entry_count, sink);
        for (const auto& entry : entries) {
            writeTokenIds(output, entry.token_ids, field, sink);
            writeGoalSpan(output, entry.span, sink);
        }
    };

    if (!sequence.concept_block_spans) {
        writeScalar(output, std::uint32_t{0}, sink);
        writeScalar(output, std::uint32_t{0}, sink);
        return;
    }
    write_entries(sequence.concept_block_spans->knowns, "knowns");
    write_entries(sequence.concept_block_spans->unknowns, "unknowns");
}

std::shared_ptr<const GRIM::ConceptBlockSpans>
readConceptBlockSpansForSequence(
    std::istream& input,
    const std::string& source) {
    auto spans = std::make_shared<GRIM::ConceptBlockSpans>();
    auto read_entries = [&input, &source](
        std::vector<GRIM::ConceptBlockSpanEntry>& entries) {
        const std::uint32_t entry_count =
            readScalar<std::uint32_t>(input, source);
        entries.reserve(entry_count);
        for (std::uint32_t index = 0; index < entry_count; ++index) {
            GRIM::ConceptBlockSpanEntry entry;
            entry.token_ids = readTokenIds(input, source);
            entry.span = readGoalSpan(input, source);
            entries.push_back(std::move(entry));
        }
    };
    read_entries(spans->knowns);
    read_entries(spans->unknowns);
    if (spans->empty()) {
        return nullptr;
    }
    std::shared_ptr<const GRIM::ConceptBlockSpans> immutable_spans =
        std::move(spans);
    validateConceptBlockSpanMetadata(immutable_spans, source);
    return immutable_spans;
}

void publishTempFileOrThrow(const fs::path& temp_path, const fs::path& final_path) {
#ifdef _WIN32
    const std::wstring temp_w = temp_path.wstring();
    const std::wstring final_w = final_path.wstring();
    if (!MoveFileExW(temp_w.c_str(), final_w.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        throw std::runtime_error("[GRMT] failed to publish temp file " + temp_path.string() +
                                 " -> " + final_path.string() + " (GetLastError=" +
                                 std::to_string(static_cast<unsigned long>(error)) + ")");
    }
#else
    std::error_code ec;
    fs::rename(temp_path, final_path, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to publish temp file " + temp_path.string() +
                                 " -> " + final_path.string() + ": " + ec.message());
    }
#endif
}

bool shouldWriteSequence(const GrmtSequence& sequence) {
    return !sequence.token_ids.empty() && sequence.hasAnyValidTarget();
}

void writeAtomTableForSequence(std::ostream& output,
                               const GrmtSequence& sequence,
                               const std::string& sink) {
    const std::uint8_t has_atom_table = sequence.atom_table ? 1 : 0;
    writeScalar(output, has_atom_table, sink);
    if (has_atom_table == 0) {
        return;
    }

    sequence.atom_table->serializeToStreamOrThrow(output, sink.c_str());
    if (!output) {
        throw std::runtime_error("[GRMT] failed to serialize AtomTable to " + sink);
    }
}

std::shared_ptr<GRIM::Tokenizer::AtomTable> readAtomTableForSequence(std::istream& input,
                                                                     const std::string& source) {
    const std::uint8_t has_atom_table = readScalar<std::uint8_t>(input, source);
    if (has_atom_table > 1) {
        throw std::runtime_error("[GRMT] invalid atom_table flag in " + source +
                                 ": " + std::to_string(static_cast<unsigned int>(has_atom_table)));
    }
    if (has_atom_table == 0) {
        return nullptr;
    }

    auto atom_table = std::make_shared<GRIM::Tokenizer::AtomTable>();
    atom_table->deserializeFromStreamOrThrow(input, source.c_str());
    return atom_table;
}

void writeLocalAtomTableForSequence(std::ostream& output,
                                    const GrmtSequence& sequence,
                                    const std::string& sink) {
    const std::uint8_t has_local_atom_table = sequence.local_atom_table ? 1 : 0;
    writeScalar(output, has_local_atom_table, sink);
    if (has_local_atom_table == 0) {
        return;
    }
    sequence.local_atom_table->serializeToStreamOrThrow(output, sink.c_str());
}

std::shared_ptr<GRIM::Tokenizer::SequenceLocalAtomTable>
readLocalAtomTableForSequence(std::istream& input, const std::string& source) {
    const std::uint8_t has_local_atom_table = readScalar<std::uint8_t>(input, source);
    if (has_local_atom_table > 1) {
        throw std::runtime_error("[GRMT] invalid local_atom_table flag in " + source +
                                 ": " + std::to_string(
                                     static_cast<unsigned int>(has_local_atom_table)));
    }
    if (has_local_atom_table == 0) {
        return nullptr;
    }
    auto table = std::make_shared<GRIM::Tokenizer::SequenceLocalAtomTable>();
    table->deserializeFromStreamOrThrow(input, source.c_str());
    return table;
}

} // namespace

bool GrmtSequence::hasAnyValidTarget() const {
    return std::any_of(targets.begin(), targets.end(), [](int target) { return target >= 0; });
}

void GrmtSequence::validateForWrite(const std::string& source) const {
    const std::size_t n = token_ids.size();
    if (!token_atom_aux_target_mask.empty()) {
        throw std::runtime_error(
            "[GRMT] " + source +
            ": token_atom_aux_target_mask is runtime-derived and must not be serialized");
    }
    if (targets.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": targets.size()=" +
                                 std::to_string(targets.size()) + " != token_ids.size()=" +
                                 std::to_string(n));
    }
    if (token_numeric_values.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_numeric_values.size()=" +
                                 std::to_string(token_numeric_values.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_atom_mask.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_atom_mask.size()=" +
                                 std::to_string(token_atom_mask.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_atom_flags.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_atom_flags.size()=" +
                                 std::to_string(token_atom_flags.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (atom_entry_ids.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": atom_entry_ids.size()=" +
                                 std::to_string(atom_entry_ids.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_local_atom_indices.size() != n) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": token_local_atom_indices.size()=" +
                                 std::to_string(token_local_atom_indices.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (token_exec_slot_indices.size() != n) {
        throw std::runtime_error("[GRMT] " + source + ": token_exec_slot_indices.size()=" +
                                 std::to_string(token_exec_slot_indices.size()) +
                                 " != token_ids.size()=" + std::to_string(n));
    }
    if (!GRIM::Execution::isValidExecutionGateTarget(execution_gate_target)) {
        throw std::runtime_error("[GRMT] " + source + ": invalid execution_gate_target");
    }
    const bool gate_supervised =
        execution_gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
    if (gate_supervised && prompt_length <= 0) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": supervised execution gate requires a non-empty prompt span");
    }
    if (prompt_length == 0) {
        if (prompt_end_pos != -1) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": empty prompt span requires prompt_end_pos=-1");
        }
    } else {
        if (prompt_length < 0 || prompt_end_pos < 0 ||
            prompt_end_pos >= static_cast<std::int32_t>(n)) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": prompt span is outside the sequence");
        }
        const std::int32_t prompt_start_pos = prompt_end_pos - prompt_length + 1;
        if (prompt_start_pos < 0) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": prompt_length extends before the sequence start");
        }
    }
    if (execution_active &&
        execution_gate_target != GRIM::Execution::ExecutionGateTarget::EXECUTE) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": execution-active row must carry EXECUTE gate target");
    }
    if (!execution_active &&
        execution_gate_target == GRIM::Execution::ExecutionGateTarget::EXECUTE) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": EXECUTE gate target requires execution_active=true");
    }

    bool saw_atom_entry = false;
    bool saw_local_atom = false;
    for (std::size_t i = 0; i < n; ++i) {
        const bool token_is_atom_open = GRIM::Tokenizer::isAtomOpenTokenId(token_ids[i]);
        if (token_atom_mask[i] > 1) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": token_atom_mask must be binary at index=" +
                                     std::to_string(i) + " value=" +
                                     std::to_string(token_atom_mask[i]));
        }

        if (token_is_atom_open) {
            if (token_atom_mask[i] != 1) {
                throw std::runtime_error("[GRMT] " + source +
                                         ": atom opening boundary has token_atom_mask=0 at index=" +
                                         std::to_string(i) + " token_id=" +
                                         std::to_string(token_ids[i]));
            }
            if (atom_entry_ids[i] == GRIM::Tokenizer::kAtomEntryNone) {
                throw std::runtime_error("[GRMT] " + source +
                                         ": atom opening boundary has no atom_entry_id at index=" +
                                         std::to_string(i) + " token_id=" +
                                         std::to_string(token_ids[i]));
            }
            const auto token_type = GRIM::Tokenizer::tokenIdToAtomType(token_ids[i]);
            if (token_local_atom_indices[i] == GRIM::Tokenizer::kLocalAtomIndexNone) {
                throw std::runtime_error("[GRMT] " + source +
                                         ": atom opening boundary has no local atom index at index=" +
                                         std::to_string(i));
            }
            if (!local_atom_table ||
                !local_atom_table->contains(token_type, token_local_atom_indices[i])) {
                throw std::runtime_error("[GRMT] " + source +
                                         ": atom opening boundary has invalid local address (" +
                                         std::string(GRIM::Tokenizer::atomTypeName(token_type)) +
                                         ", " + std::to_string(token_local_atom_indices[i]) +
                                         ") at token index=" + std::to_string(i));
            }
            saw_atom_entry = true;
            saw_local_atom = true;
            continue;
        }

        if (token_atom_mask[i] != 0 ||
            atom_entry_ids[i] != GRIM::Tokenizer::kAtomEntryNone ||
            token_local_atom_indices[i] != GRIM::Tokenizer::kLocalAtomIndexNone ||
            token_numeric_values[i] != 0.0f ||
            token_atom_flags[i] != 0) {
            const char* position_kind = GRIM::Tokenizer::isAtomCloseTokenId(token_ids[i])
                ? "atom closing boundary"
                : "non-opening token";
            throw std::runtime_error("[GRMT] " + source + ": " + position_kind +
                                     " carries atom metadata at index=" +
                                     std::to_string(i) + " token_id=" +
                                     std::to_string(token_ids[i]));
        }
    }

    if (saw_atom_entry && !atom_table) {
        throw std::runtime_error("[GRMT] " + source + ": atom_entry_ids are present but atom_table is null");
    }
    if (saw_local_atom && !local_atom_table) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": local atom indices are present but local_atom_table is null");
    }
    if (atom_table) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t entry_id = atom_entry_ids[i];
            if (entry_id == GRIM::Tokenizer::kAtomEntryNone) {
                continue;
            }
            const auto entry = atom_table->getAtom(entry_id);
            if (!entry.has_value()) {
                throw std::runtime_error("[GRMT] " + source + ": atom_entry_id=" +
                                         std::to_string(entry_id) +
                                         " is not retrievable from atom_table at token index=" +
                                         std::to_string(i));
            }
            const auto token_type = GRIM::Tokenizer::tokenIdToAtomType(token_ids[i]);
            if (entry->type != token_type) {
                throw std::runtime_error("[GRMT] " + source + ": atom opening boundary type=" +
                                         std::string(GRIM::Tokenizer::atomTypeName(token_type)) +
                                         " does not match atom_entry_id=" +
                                         std::to_string(entry_id) + " type=" +
                                         GRIM::Tokenizer::atomTypeName(entry->type) +
                                         " at token index=" + std::to_string(i));
            }
        }
    }

    if (execution_active && transition_targets.empty()) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": execution_active sequence has no transition_targets");
    }
    if (execution_active && compiled_slot_bindings.empty()) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": execution_active sequence has no compiled_slot_bindings");
    }
    if (execution_active && compiled_transition_bindings.empty()) {
        throw std::runtime_error(
            "[GRMT] " + source +
            ": execution_active sequence has no compiled_transition_bindings");
    }
    if (execution_active && compiled_bootstrap_bindings.empty()) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": execution_active sequence has no compiled_bootstrap_bindings");
    }
    if (!execution_active &&
        (!compiled_slot_bindings.empty() ||
         !compiled_transition_bindings.empty() ||
         !compiled_bootstrap_bindings.empty() ||
         !transition_targets.empty())) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": inactive sequence carries execution metadata");
    }

    validateGoalMetadata(goal, source);
    validateGoalTokenSlices(goal, token_ids, source);
    validateConceptBlockSpanMetadata(concept_block_spans, source);
    validateConceptBlockSpanTokenSlices(
        concept_block_spans, token_ids, source);

    std::unordered_set<std::uint64_t> slot_ids;
    std::unordered_set<std::int32_t> slot_indices;
    for (const auto& binding : compiled_slot_bindings) {
        if (!binding.slot_id.valid() || !binding.slot_index.valid()) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": compiled slot binding contains an invalid primitive");
        }
        if (static_cast<std::size_t>(binding.slot_index.dense()) >=
            compiled_slot_bindings.size()) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": compiled SlotIndex is outside its dense table");
        }
        if (!slot_ids.insert(binding.slot_id.serialized()).second ||
            !slot_indices.insert(binding.slot_index.dense()).second) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": compiled slot bindings are not bijective");
        }
    }

    std::unordered_set<std::uint64_t> transition_ids;
    std::unordered_set<std::int32_t> transition_indices;
    for (const auto& binding : compiled_transition_bindings) {
        if (!binding.transition_id.valid() ||
            !binding.transition_index.valid()) {
            throw std::runtime_error(
                "[GRMT] " + source +
                ": compiled transition binding contains an invalid primitive");
        }
        if (static_cast<std::size_t>(binding.transition_index.dense()) >=
            compiled_transition_bindings.size()) {
            throw std::runtime_error(
                "[GRMT] " + source +
                ": compiled TransitionIndex is outside its dense table");
        }
        if (!transition_ids.insert(
                binding.transition_id.serialized()).second ||
            !transition_indices.insert(
                binding.transition_index.dense()).second) {
            throw std::runtime_error(
                "[GRMT] " + source +
                ": compiled transition bindings are not bijective");
        }
    }

    std::vector<std::int32_t> expected_slot_indices(n, -1);
    std::unordered_set<std::int32_t> bootstrap_positions;
    std::unordered_set<std::int32_t> bootstrap_slot_indices;
    for (std::size_t i = 0; i < compiled_bootstrap_bindings.size(); ++i) {
        const auto& binding = compiled_bootstrap_bindings[i];
        if (binding.binding_id != static_cast<std::int32_t>(i)) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": bootstrap binding_id is not its row-local ordinal");
        }
        if (binding.token_pos < 0 ||
            static_cast<std::size_t>(binding.token_pos) >= n) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": bootstrap token_pos is outside the sequence");
        }
        if (!GRIM::Execution::findSlotId(
                compiled_slot_bindings, binding.slot_index).has_value()) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": bootstrap SlotIndex has no semantic binding");
        }
        if (!bootstrap_positions.insert(binding.token_pos).second ||
            !bootstrap_slot_indices.insert(binding.slot_index.dense()).second) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": bootstrap bindings are not injective");
        }
        expected_slot_indices[static_cast<std::size_t>(binding.token_pos)] =
            binding.slot_index.dense();
    }
    if (token_exec_slot_indices != expected_slot_indices) {
        throw std::runtime_error("[GRMT] " + source +
                                 ": token_exec_slot_indices do not match bootstrap bindings");
    }
    for (const auto& invocation : transition_targets) {
        if (!GRIM::Execution::findTransitionIndex(
                compiled_transition_bindings,
                invocation.transition_id).has_value()) {
            throw std::runtime_error("[GRMT] " + source +
                                     ": TransitionId has no compiled runtime binding");
        }
        for (const auto slot : invocation.arguments) {
            if (!GRIM::Execution::findSlotIndex(
                    compiled_slot_bindings, slot).has_value()) {
                throw std::runtime_error(
                    "[GRMT] " + source +
                    ": transition argument SlotId has no compiled runtime binding");
            }
        }
        for (const auto slot : invocation.results) {
            if (!GRIM::Execution::findSlotIndex(
                    compiled_slot_bindings, slot).has_value()) {
                throw std::runtime_error(
                    "[GRMT] " + source +
                    ": transition result SlotId has no compiled runtime binding");
            }
        }
    }
}

GrmtCorpusWriter::GrmtCorpusWriter(
    const fs::path& final_path,
    std::uint32_t num_sequences,
    std::uint32_t vocab_size)
    : final_path_(final_path)
    , temp_path_(final_path)
    , expected_sequences_(num_sequences)
    , vocab_size_(vocab_size)
{
    pathString(final_path_);
    if (num_sequences == 0) {
        throw std::runtime_error("[GRMT] writer requires num_sequences > 0 for " + final_path_.string());
    }
    if (vocab_size == 0) {
        throw std::runtime_error("[GRMT] writer requires vocab_size > 0 for " + final_path_.string());
    }

    ensureParentDirectory(final_path_);
    temp_path_ += ".tmp";
    std::error_code ec;
    fs::remove(temp_path_, ec);
    if (ec) {
        throw std::runtime_error("[GRMT] failed to remove stale temp file " + temp_path_.string() +
                                 ": " + ec.message());
    }

    file_.open(temp_path_, std::ios::binary | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] cannot open temp file for write: " + temp_path_.string());
    }

    const GRIM::GRMT::Header header = GRIM::GRMT::makeCurrentHeader(num_sequences, vocab_size);
    GRIM::GRMT::writeHeaderOrThrow(file_, header, temp_path_.string());
}

GrmtCorpusWriter::~GrmtCorpusWriter() noexcept {
    if (file_.is_open()) {
        file_.close();
    }
    if (!committed_) {
        std::error_code ec;
        fs::remove(temp_path_, ec);
    }
}

void GrmtCorpusWriter::writeSequence(const GrmtSequence& sequence) {
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] writeSequence called after stream close: " + temp_path_.string());
    }
    if (written_sequences_ >= expected_sequences_) {
        throw std::runtime_error("[GRMT] writeSequence would exceed header num_sequences=" +
                                 std::to_string(expected_sequences_) + " for " + temp_path_.string());
    }

    const std::string sink = temp_path_.string() + "#seq" + std::to_string(written_sequences_);
    sequence.validateForWrite(sink);
    validateSequenceTokenRange(sequence, vocab_size_, sink);
    validateGoalTokenRange(sequence.goal, vocab_size_, sink);
    validateConceptBlockSpanTokenRange(
        sequence.concept_block_spans, vocab_size_, sink);

    const std::uint32_t len = static_cast<std::uint32_t>(sequence.token_ids.size());
    writeScalar(file_, len, sink);
    writeExact(file_, sequence.token_ids.data(), static_cast<std::size_t>(len) * sizeof(int), sink);
    writeExact(file_, sequence.targets.data(), static_cast<std::size_t>(len) * sizeof(int), sink);
    writeExact(file_, sequence.token_numeric_values.data(), static_cast<std::size_t>(len) * sizeof(float), sink);
    writeExact(file_, sequence.token_atom_mask.data(), static_cast<std::size_t>(len) * sizeof(std::uint8_t), sink);
    writeExact(file_, sequence.token_atom_flags.data(), static_cast<std::size_t>(len) * sizeof(std::uint32_t), sink);
    writeExact(file_, sequence.atom_entry_ids.data(), static_cast<std::size_t>(len) * sizeof(std::uint32_t), sink);
    writeAtomTableForSequence(file_, sequence, sink);
    writeExact(file_, sequence.token_local_atom_indices.data(),
               static_cast<std::size_t>(len) * sizeof(std::uint32_t), sink);
    writeLocalAtomTableForSequence(file_, sequence, sink);

    const std::uint8_t exec_active = sequence.execution_active ? 1 : 0;
    writeScalar(file_, exec_active, sink);
    const std::int8_t gate_target = static_cast<std::int8_t>(sequence.execution_gate_target);
    writeScalar(file_, gate_target, sink);
    writeScalar(file_, sequence.prompt_end_pos, sink);
    writeScalar(file_, sequence.prompt_length, sink);
    writeGoalForSequence(file_, sequence, sink);
    writeConceptBlockSpansForSequence(file_, sequence, sink);
    writeExact(file_, sequence.token_exec_slot_indices.data(), static_cast<std::size_t>(len) * sizeof(std::int32_t), sink);

    const std::uint32_t csb_count =
        static_cast<std::uint32_t>(sequence.compiled_slot_bindings.size());
    writeScalar(file_, csb_count, sink);
    for (const auto& binding : sequence.compiled_slot_bindings) {
        writeSlotId(file_, binding.slot_id, sink);
        writeSlotIndex(file_, binding.slot_index, sink);
    }

    const std::uint32_t ctb_count =
        static_cast<std::uint32_t>(
            sequence.compiled_transition_bindings.size());
    writeScalar(file_, ctb_count, sink);
    for (const auto& binding : sequence.compiled_transition_bindings) {
        writeTransitionId(file_, binding.transition_id, sink);
        writeTransitionIndex(file_, binding.transition_index, sink);
    }

    const std::uint32_t cbb_count = static_cast<std::uint32_t>(sequence.compiled_bootstrap_bindings.size());
    writeScalar(file_, cbb_count, sink);
    for (const auto& binding : sequence.compiled_bootstrap_bindings) {
        writeScalar(file_, binding.binding_id, sink);
        writeScalar(file_, binding.token_pos, sink);
        writeSlotIndex(file_, binding.slot_index, sink);
    }

    const std::uint32_t target_count =
        static_cast<std::uint32_t>(sequence.transition_targets.size());
    writeScalar(file_, target_count, sink);
    for (const auto& invocation : sequence.transition_targets) {
        writeTransitionId(file_, invocation.transition_id, sink);
        const std::uint32_t argument_count =
            static_cast<std::uint32_t>(invocation.arguments.size());
        writeScalar(file_, argument_count, sink);
        for (const GRIM::Execution::SlotId slot : invocation.arguments) {
            writeSlotId(file_, slot, sink);
        }
        const std::uint32_t result_count =
            static_cast<std::uint32_t>(invocation.results.size());
        writeScalar(file_, result_count, sink);
        for (const GRIM::Execution::SlotId slot : invocation.results) {
            writeSlotId(file_, slot, sink);
        }
    }

    ++written_sequences_;
}

GrmtSaveReport GrmtCorpusWriter::commit(std::size_t dropped_empty_sequences,
                                        std::size_t dropped_targetless_sequences) {
    if (written_sequences_ != expected_sequences_) {
        throw std::runtime_error("[GRMT] commit refused for " + temp_path_.string() +
                                 ": written_sequences=" + std::to_string(written_sequences_) +
                                 " != expected_sequences=" + std::to_string(expected_sequences_));
    }
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] commit called after stream close: " + temp_path_.string());
    }

    file_.flush();
    file_.close();
    if (!file_.good()) {
        throw std::runtime_error("[GRMT] stream failed on flush/close: " + temp_path_.string());
    }

    publishTempFileOrThrow(temp_path_, final_path_);
    committed_ = true;

    GrmtSaveReport report{};
    report.path = final_path_;
    report.written_sequences = written_sequences_;
    report.vocab_size = vocab_size_;
    report.dropped_empty_sequences = dropped_empty_sequences;
    report.dropped_targetless_sequences = dropped_targetless_sequences;
    return report;
}

GrmtCorpusReader::GrmtCorpusReader(const fs::path& path)
    : path_(path),
      file_buffer_(kReadBufferBytes)
{
    pathString(path_);
    if (file_.rdbuf()->pubsetbuf(
            file_buffer_.data(),
            static_cast<std::streamsize>(file_buffer_.size())) == nullptr) {
        throw std::runtime_error("[GRMT] failed to configure read buffer for: " + path_.string());
    }
    file_.open(path_, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("[GRMT] cannot open file for read: " + path_.string());
    }
    header_ = GRIM::GRMT::readHeaderOrThrow(file_, path_.string());
}

GrmtCorpusReader::~GrmtCorpusReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

bool GrmtCorpusReader::readNext(GrmtSequence& out_sequence) {
    if (sequences_read_ >= header_.num_sequences) {
        return false;
    }

    const std::string source = path_.string() + "#seq" + std::to_string(sequences_read_);
    const std::uint32_t seq_len = readScalar<std::uint32_t>(file_, source);
    if (seq_len == 0) {
        throw std::runtime_error("[GRMT] sequence length is zero in " + source);
    }

    GrmtSequence seq;
    seq.token_ids.resize(seq_len);
    seq.targets.resize(seq_len);
    seq.token_numeric_values.resize(seq_len);
    seq.token_atom_mask.resize(seq_len);
    seq.token_atom_flags.resize(seq_len);
    seq.atom_entry_ids.resize(seq_len);
    seq.token_local_atom_indices.resize(seq_len);

    readExact(file_, seq.token_ids.data(), static_cast<std::size_t>(seq_len) * sizeof(int), source);
    readExact(file_, seq.targets.data(), static_cast<std::size_t>(seq_len) * sizeof(int), source);
    readExact(file_, seq.token_numeric_values.data(), static_cast<std::size_t>(seq_len) * sizeof(float), source);
    readExact(file_, seq.token_atom_mask.data(), static_cast<std::size_t>(seq_len) * sizeof(std::uint8_t), source);
    readExact(file_, seq.token_atom_flags.data(), static_cast<std::size_t>(seq_len) * sizeof(std::uint32_t), source);
    readExact(file_, seq.atom_entry_ids.data(), static_cast<std::size_t>(seq_len) * sizeof(std::uint32_t), source);
    seq.atom_table = readAtomTableForSequence(file_, source);
    readExact(file_, seq.token_local_atom_indices.data(),
              static_cast<std::size_t>(seq_len) * sizeof(std::uint32_t), source);
    seq.local_atom_table = readLocalAtomTableForSequence(file_, source);

    const std::uint8_t exec_active = readScalar<std::uint8_t>(file_, source);
    seq.execution_active = (exec_active != 0);
    const std::int8_t gate_target = readScalar<std::int8_t>(file_, source);
    seq.execution_gate_target = static_cast<GRIM::Execution::ExecutionGateTarget>(gate_target);
    seq.prompt_end_pos = readScalar<std::int32_t>(file_, source);
    seq.prompt_length = readScalar<std::int32_t>(file_, source);
    seq.goal = readGoalForSequence(file_, source);
    seq.concept_block_spans =
        readConceptBlockSpansForSequence(file_, source);

    seq.token_exec_slot_indices.resize(seq_len);
    readExact(file_, seq.token_exec_slot_indices.data(), static_cast<std::size_t>(seq_len) * sizeof(std::int32_t), source);

    const std::uint32_t csb_count = readScalar<std::uint32_t>(file_, source);
    seq.compiled_slot_bindings.reserve(csb_count);
    for (std::uint32_t i = 0; i < csb_count; ++i) {
        seq.compiled_slot_bindings.push_back(GRIM::Execution::CompiledSlotBinding{
            readSlotId(file_, source),
            readSlotIndex(file_, source)});
    }

    const std::uint32_t ctb_count = readScalar<std::uint32_t>(file_, source);
    seq.compiled_transition_bindings.reserve(ctb_count);
    for (std::uint32_t i = 0; i < ctb_count; ++i) {
        seq.compiled_transition_bindings.push_back(
            GRIM::Execution::CompiledTransitionBinding{
                readTransitionId(file_, source),
                readTransitionIndex(file_, source)});
    }

    const std::uint32_t cbb_count = readScalar<std::uint32_t>(file_, source);
    seq.compiled_bootstrap_bindings.reserve(cbb_count);
    for (std::uint32_t i = 0; i < cbb_count; ++i) {
        seq.compiled_bootstrap_bindings.push_back(
            GRIM::Execution::CompiledBootstrapBinding{
                readScalar<std::int32_t>(file_, source),
                readScalar<std::int32_t>(file_, source),
                readSlotIndex(file_, source)});
    }

    const std::uint32_t target_count =
        readScalar<std::uint32_t>(file_, source);
    seq.transition_targets.reserve(target_count);
    for (std::uint32_t i = 0; i < target_count; ++i) {
        GRIM::Execution::TransitionInvocation invocation;
        invocation.transition_id = readTransitionId(file_, source);
        const std::uint32_t argument_count =
            readScalar<std::uint32_t>(file_, source);
        invocation.arguments.reserve(argument_count);
        for (std::uint32_t argument = 0;
             argument < argument_count;
             ++argument) {
            invocation.arguments.push_back(readSlotId(file_, source));
        }
        const std::uint32_t result_count =
            readScalar<std::uint32_t>(file_, source);
        invocation.results.reserve(result_count);
        for (std::uint32_t result = 0;
             result < result_count;
             ++result) {
            invocation.results.push_back(readSlotId(file_, source));
        }
        seq.transition_targets.push_back(std::move(invocation));
    }

    seq.validateForWrite(source);
    validateSequenceTokenRange(seq, header_.vocab_size, source);
    validateGoalTokenRange(seq.goal, header_.vocab_size, source);
    validateConceptBlockSpanTokenRange(
        seq.concept_block_spans, header_.vocab_size, source);

    out_sequence = std::move(seq);
    ++sequences_read_;
    return true;
}

GrmtCorpus GrmtCorpusReader::readAll() {
    GrmtCorpus corpus;
    corpus.header = header_;
    corpus.sequences.reserve(header_.num_sequences);

    GrmtSequence sequence;
    while (readNext(sequence)) {
        corpus.sequences.push_back(std::move(sequence));
        sequence = GrmtSequence{};
    }

    if (corpus.sequences.size() != header_.num_sequences) {
        throw std::runtime_error("[GRMT] loaded sequence count mismatch in " + path_.string() +
                                 ": loaded=" + std::to_string(corpus.sequences.size()) +
                                 " header=" + std::to_string(header_.num_sequences));
    }
    return corpus;
}

GRIM::GRMT::Header loadGrmtHeader(const fs::path& path) {
    return GRIM::GRMT::readHeaderOrThrow(pathString(path));
}

GrmtSaveReport saveGrmtCorpus(
    const fs::path& path,
    const std::vector<GrmtSequence>& sequences,
    std::uint32_t vocab_size) {
    pathString(path);
    if (vocab_size == 0) {
        throw std::runtime_error("[GRMT] saveGrmtCorpus requires vocab_size > 0 for " + path.string());
    }

    std::uint32_t valid_sequences = 0;
    std::size_t dropped_empty = 0;
    std::size_t dropped_targetless = 0;
    for (const auto& sequence : sequences) {
        if (sequence.token_ids.empty()) {
            ++dropped_empty;
            continue;
        }
        if (!sequence.hasAnyValidTarget()) {
            ++dropped_targetless;
            continue;
        }
        ++valid_sequences;
    }

    if (valid_sequences == 0) {
        throw std::runtime_error("[GRMT] refusing to write zero valid sequences to " + path.string() +
                                 " (input=" + std::to_string(sequences.size()) +
                                 ", dropped_empty=" + std::to_string(dropped_empty) +
                                 ", dropped_targetless=" + std::to_string(dropped_targetless) + ")");
    }
    if (dropped_empty > 0 || dropped_targetless > 0) {
        throw std::runtime_error("[GRMT] refused writer-side drops for " + path.string() +
                                 " (dropped_empty=" + std::to_string(dropped_empty) +
                                 ", dropped_targetless=" + std::to_string(dropped_targetless) + ")");
    }

    GrmtCorpusWriter writer(path, valid_sequences, vocab_size);
    for (const auto& sequence : sequences) {
        if (!shouldWriteSequence(sequence)) {
            continue;
        }
        writer.writeSequence(sequence);
    }
    return writer.commit(dropped_empty, dropped_targetless);
}

GrmtCorpus loadGrmtCorpus(const fs::path& path) {
    GrmtCorpusReader reader(path);
    return reader.readAll();
}

} // namespace GRIM::TokenizerArtifacts
