//======================================================//
//  Startup/ConceptSupervision.cu
//
//  ConceptBlock field-policy projection applied to complete GRMT rows
//  before boundary injection and sliding-window construction.
//======================================================//

#include "ConceptSupervision.hpp"

#include "../../../Shared/ConceptBlock/ConceptBlockSpans.hpp"
#include "../../../Shared/ConceptBlock/NamedConceptSpans.hpp"
#include "../../../Shared/Goal/Goal.hpp"
#include "../../../Shared/TokenizerArtifacts/GrmtSequence.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIMText::Training {

namespace {

using GrmtSequence = GRIM::TokenizerArtifacts::GrmtSequence;

std::optional<GRIM::GoalTokenSpan> namedConceptFieldSpan(
    const GrmtSequence& sequence,
    ConceptField field) {
    if (!sequence.named_concept_spans) return std::nullopt;
    const auto* entry = sequence.named_concept_spans->find(
        conceptFieldName(field));
    return entry ? std::optional<GRIM::GoalTokenSpan>{entry->span}
                 : std::nullopt;
}

std::optional<GRIM::GoalTokenSpan> conceptFieldSpan(
    const GrmtSequence& sequence,
    ConceptField field) {
    switch (field) {
        case ConceptField::Prompt:
            if (sequence.prompt_length <= 0 || sequence.prompt_end_pos < 0) return std::nullopt;
            return GRIM::GoalTokenSpan{
                sequence.prompt_end_pos - sequence.prompt_length + 1,
                sequence.prompt_end_pos + 1};
        case ConceptField::TargetState:
            if (sequence.goal && sequence.goal->target_state)
                return sequence.goal->target_state->span;
            return std::nullopt;
        case ConceptField::SuccessCriteriaAndEvidence:
            if (sequence.goal && sequence.goal->success_criteria)
                return sequence.goal->success_criteria->span;
            return std::nullopt;
        case ConceptField::Constraints:
            if (sequence.goal && sequence.goal->constraints)
                return sequence.goal->constraints->span;
            return std::nullopt;
        case ConceptField::Knowns:
            if (sequence.concept_block_spans &&
                !sequence.concept_block_spans->knowns.empty()) {
                return GRIM::GoalTokenSpan{
                    sequence.concept_block_spans->knowns.front().span.begin,
                    sequence.concept_block_spans->knowns.back().span.end};
            }
            return std::nullopt;
        case ConceptField::Unknowns:
            if (sequence.concept_block_spans &&
                !sequence.concept_block_spans->unknowns.empty()) {
                return GRIM::GoalTokenSpan{
                    sequence.concept_block_spans->unknowns.front().span.begin,
                    sequence.concept_block_spans->unknowns.back().span.end};
            }
            return std::nullopt;
        case ConceptField::Reasoning:
        case ConceptField::Determine:
        case ConceptField::Define:
        case ConceptField::Execute:
        case ConceptField::Update:
        case ConceptField::Answer:
            return namedConceptFieldSpan(sequence, field);
        case ConceptField::Count:
            break;
    }
    return std::nullopt;
}

void retainMetadataThrough(GrmtSequence& sequence, std::int32_t cut) {
    if (sequence.goal) {
        auto retained = std::make_shared<GRIM::Goal>();
        if (sequence.goal->target_state.has_value() &&
            sequence.goal->target_state->span.end <= cut) {
            retained->target_state = sequence.goal->target_state;
        }
        if (sequence.goal->success_criteria.has_value() &&
            sequence.goal->success_criteria->span.end <= cut) {
            retained->success_criteria = sequence.goal->success_criteria;
        }
        if (sequence.goal->constraints.has_value() &&
            sequence.goal->constraints->span.end <= cut) {
            retained->constraints = sequence.goal->constraints;
        }
        if (!retained->target_state && !retained->success_criteria &&
            !retained->constraints) {
            sequence.goal.reset();
        } else {
            sequence.goal = std::move(retained);
        }
    }
    if (sequence.concept_block_spans) {
        auto retained = std::make_shared<GRIM::ConceptBlockSpans>();
        const auto copy_fitting = [cut](
            const std::vector<GRIM::ConceptBlockSpanEntry>& source,
            std::vector<GRIM::ConceptBlockSpanEntry>& destination) {
            for (const auto& entry : source) {
                if (entry.span.end <= cut) destination.push_back(entry);
            }
        };
        copy_fitting(sequence.concept_block_spans->knowns, retained->knowns);
        copy_fitting(sequence.concept_block_spans->unknowns, retained->unknowns);
        const auto copy_optional = [cut](
            const std::optional<GRIM::ConceptBlockSpanEntry>& source_entry,
            std::optional<GRIM::ConceptBlockSpanEntry>& destination) {
            if (source_entry && source_entry->span.end <= cut) {
                destination = source_entry;
            }
        };
        copy_optional(sequence.concept_block_spans->reasoning, retained->reasoning);
        copy_optional(sequence.concept_block_spans->determine, retained->determine);
        copy_optional(sequence.concept_block_spans->define, retained->define);
        copy_optional(sequence.concept_block_spans->execute, retained->execute);
        copy_optional(sequence.concept_block_spans->update, retained->update);
        if (retained->empty()) sequence.concept_block_spans.reset();
        else sequence.concept_block_spans = std::move(retained);
    }
    if (sequence.named_concept_spans) {
        auto retained = std::make_shared<GRIM::NamedConceptSpans>();
        retained->entries.reserve(sequence.named_concept_spans->entries.size());
        for (const auto& entry : sequence.named_concept_spans->entries) {
            if (entry.span.end <= cut) retained->entries.push_back(entry);
        }
        if (retained->empty()) sequence.named_concept_spans.reset();
        else sequence.named_concept_spans = std::move(retained);
    }
}

} // namespace

const char* conceptFieldName(ConceptField field) {
    switch (field) {
        case ConceptField::Prompt: return "prompt";
        case ConceptField::TargetState: return "target_state";
        case ConceptField::SuccessCriteriaAndEvidence:
            return "success_criteria_and_evidence";
        case ConceptField::Constraints: return "constraints";
        case ConceptField::Knowns: return "knowns";
        case ConceptField::Unknowns: return "unknowns";
        case ConceptField::Reasoning: return "reasoning";
        case ConceptField::Determine: return "determine";
        case ConceptField::Define: return "define";
        case ConceptField::Execute: return "execute";
        case ConceptField::Update: return "update";
        case ConceptField::Answer: return "answer";
        case ConceptField::Count: break;
    }
    return "unknown";
}

ConceptField parseConceptField(const std::string& name, const std::string& source) {
    if (name == "prompt") return ConceptField::Prompt;
    if (name == "target_state") return ConceptField::TargetState;
    if (name == "success_criteria_and_evidence")
        return ConceptField::SuccessCriteriaAndEvidence;
    if (name == "constraints") return ConceptField::Constraints;
    if (name == "knowns") return ConceptField::Knowns;
    if (name == "unknowns") return ConceptField::Unknowns;
    if (name == "reasoning") return ConceptField::Reasoning;
    if (name == "determine") return ConceptField::Determine;
    if (name == "define") return ConceptField::Define;
    if (name == "execute") return ConceptField::Execute;
    if (name == "update") return ConceptField::Update;
    if (name == "answer") return ConceptField::Answer;
    throw std::runtime_error(
        source + ": unknown concept field '" + name + "'");
}

bool projectConceptSupervision(
    GrmtSequence& sequence,
    const std::vector<ConceptField>& supervised_fields,
    const std::vector<ConceptField>& unsupervised_fields,
    const std::string& source) {
    std::vector<GRIM::GoalTokenSpan> supervised_spans;
    std::size_t cut = 0;
    const auto collect = [&](const std::vector<ConceptField>& fields,
                             bool supervised) {
        for (const auto field : fields) {
            const auto span = conceptFieldSpan(sequence, field);
            if (!span) continue;
            if (!span->valid() || span->begin < 0 ||
                static_cast<std::size_t>(span->end) > sequence.token_ids.size()) {
                throw std::runtime_error(
                    source + ": invalid span for field " +
                    conceptFieldName(field));
            }
            cut = std::max(cut, static_cast<std::size_t>(span->end));
            if (supervised) supervised_spans.push_back(*span);
        }
    };
    collect(supervised_fields, true);
    collect(unsupervised_fields, false);
    if (supervised_spans.empty()) {
        return false;
    }
    const auto first_supervised = std::min_element(
        supervised_spans.begin(), supervised_spans.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.begin < rhs.begin; });
    if (first_supervised->begin <= 0) {
        throw std::runtime_error(
            source + ": first supervised field has no preceding causal token");
    }
    sequence.token_ids.resize(cut);
    sequence.targets.assign(cut, -1);
    sequence.token_numeric_values.resize(cut);
    sequence.token_atom_mask.resize(cut);
    sequence.token_atom_flags.resize(cut);
    sequence.atom_entry_ids.resize(cut);
    sequence.token_local_atom_indices.resize(cut);
    if (!sequence.token_atom_aux_target_mask.empty()) {
        sequence.token_atom_aux_target_mask.resize(cut);
    }
    sequence.token_exec_slot_indices.resize(cut);
    for (const auto& span : supervised_spans) {
        for (std::int32_t position = span.begin; position < span.end; ++position) {
            sequence.targets[static_cast<std::size_t>(position - 1)] =
                sequence.token_ids[static_cast<std::size_t>(position)];
        }
    }
    sequence.prompt_length = first_supervised->begin;
    sequence.prompt_end_pos = first_supervised->begin - 1;
    retainMetadataThrough(sequence, static_cast<std::int32_t>(cut));
    sequence.answer_span.reset();
    const bool retain_execution = std::any_of(
        supervised_fields.begin(), supervised_fields.end(),
        [](const auto field) {
            return field == ConceptField::Execute ||
                   field == ConceptField::Answer;
        });
    if (!retain_execution) {
        sequence.execution_active = false;
        sequence.execution_gate_target =
            GRIM::Execution::ExecutionGateTarget::UNSUPERVISED;
        std::fill(sequence.token_exec_slot_indices.begin(),
                  sequence.token_exec_slot_indices.end(), -1);
        sequence.compiled_slot_bindings.clear();
        sequence.compiled_transition_bindings.clear();
        sequence.compiled_bootstrap_bindings.clear();
        sequence.transition_targets.clear();
    }
    return true;
}

} // namespace GRIMText::Training
