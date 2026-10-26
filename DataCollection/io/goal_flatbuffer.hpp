#pragma once
#include "../concept_block_generated.h"
#include "../../resources/models/GRIM-text/Shared/Goal/GoalContract.hpp"

namespace GRIM::GoalStorage {
inline std::string text(const flatbuffers::String* value) {
    return value ? value->str() : std::string{};
}
inline std::optional<EvidenceCondition> evidence(const GRIMConcept::EvidenceCondition* value) {
    if (!value) return std::nullopt;
    return evidenceConditionFromJson({{"predicate", text(value->predicate())},
                                      {"subject", text(value->subject())},
                                      {"target", text(value->target())}});
}
inline ConceptBlockGoal fromFlatBuffer(const GRIMConcept::Goal& source) {
    ConceptBlockGoal goal;
    goal.target_state = text(source.target_state());
    if (const auto* criteria = source.success_criteria()) {
        goal.success_criteria.reserve(criteria->size());
        for (const auto* entry : *criteria) {
            if (!entry) throw std::invalid_argument("Null success criterion");
            if (!text(entry->legacy_evidence()).empty())
                throw std::invalid_argument("Legacy prose evidence requires explicit migration; use scripts/migrate_goal_evidence.py");
            goal.success_criteria.push_back({text(entry->criterion()), evidence(entry->evidence())});
        }
    }
    if (source.legacy_constraints() && source.constraints())
        throw std::invalid_argument("Mixed legacy and structured constraints require explicit migration");
    if (const auto* legacy = source.legacy_constraints())
        for (const auto* value : *legacy) goal.constraints.push_back({text(value), std::nullopt});
    if (const auto* constraints = source.constraints())
        for (const auto* entry : *constraints) {
            if (!entry) throw std::invalid_argument("Null constraint");
            goal.constraints.push_back({text(entry->constraint()), evidence(entry->evidence())});
        }
    return goal;
}
} // namespace GRIM::GoalStorage
