#pragma once

// Goal storage/transport contract shared by authoring and inference. No corpus
// or training-entry dependency. Binding predicate semantics belongs to the
// selected environment, not this structural validator.
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
struct EvidenceCondition {
    std::string predicate, subject, target;
};
struct ConceptBlockSuccessCriterion {
    std::string criterion;
    std::optional<EvidenceCondition> evidence;
};
struct ConceptBlockConstraint {
    std::string constraint;
    std::optional<EvidenceCondition> evidence;
};
struct ConceptBlockGoal {
    std::string target_state;
    std::vector<ConceptBlockSuccessCriterion> success_criteria;
    std::vector<ConceptBlockConstraint> constraints;
};

inline void goalObjectFields(const nlohmann::json& value,
                             std::initializer_list<const char*> fields) {
    if (!value.is_object()) throw std::invalid_argument("Goal contract expects an object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool allowed = false;
        for (auto field : fields) if (it.key() == field) allowed = true;
        if (!allowed) throw std::invalid_argument("Unsupported Goal contract field: " + it.key());
    }
}
inline EvidenceCondition evidenceConditionFromJson(const nlohmann::json& value) {
    if (value.is_string())
        throw std::invalid_argument("Legacy string evidence requires explicit migration; use scripts/migrate_goal_evidence.py");
    goalObjectFields(value, {"predicate", "subject", "target"});
    EvidenceCondition condition{value.at("predicate").get<std::string>(),
                                value.at("subject").get<std::string>(),
                                value.value("target", std::string{})};
    for (const auto* text : {&condition.predicate, &condition.subject})
        if (text->find_first_not_of(" \t\r\n") == std::string::npos)
            throw std::invalid_argument("Evidence predicate and subject must be non-empty");
    return condition;
}
inline nlohmann::json evidenceConditionToJson(const EvidenceCondition& value) {
    nlohmann::json result{{"predicate", value.predicate}, {"subject", value.subject}};
    if (!value.target.empty()) result["target"] = value.target;
    (void)evidenceConditionFromJson(result);
    return result;
}
inline std::optional<EvidenceCondition> optionalEvidenceFromJson(const nlohmann::json& entry) {
    if (!entry.contains("evidence")) return std::nullopt;
    return evidenceConditionFromJson(entry.at("evidence"));
}
inline ConceptBlockGoal goalFromJson(const nlohmann::json& source) {
    goalObjectFields(source, {"target_state", "success_criteria", "constraints"});
    ConceptBlockGoal goal;
    goal.target_state = source.value("target_state", std::string{});
    const auto read = [&](const char* name, auto consume) {
        if (!source.contains(name)) return;
        const auto& entries = source.at(name);
        if (!entries.is_array()) throw std::invalid_argument(std::string(name) + " must be an array");
        for (const auto& entry : entries) consume(entry);
    };
    read("success_criteria", [&](const auto& entry) {
        goalObjectFields(entry, {"criterion", "evidence"});
        goal.success_criteria.push_back({entry.at("criterion").template get<std::string>(),
                                         optionalEvidenceFromJson(entry)});
    });
    read("constraints", [&](const auto& entry) {
        goalObjectFields(entry, {"constraint", "evidence"});
        goal.constraints.push_back({entry.at("constraint").template get<std::string>(),
                                    optionalEvidenceFromJson(entry)});
    });
    return goal;
}
inline nlohmann::json goalToJson(const ConceptBlockGoal& goal) {
    nlohmann::json result{{"target_state", goal.target_state},
                          {"success_criteria", nlohmann::json::array()},
                          {"constraints", nlohmann::json::array()}};
    for (const auto& entry : goal.success_criteria) {
        nlohmann::json item{{"criterion", entry.criterion}};
        if (entry.evidence) item["evidence"] = evidenceConditionToJson(*entry.evidence);
        result["success_criteria"].push_back(std::move(item));
    }
    for (const auto& entry : goal.constraints) {
        nlohmann::json item{{"constraint", entry.constraint}};
        if (entry.evidence) item["evidence"] = evidenceConditionToJson(*entry.evidence);
        result["constraints"].push_back(std::move(item));
    }
    return result;
}
} // namespace GRIM
