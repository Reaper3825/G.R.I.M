#pragma once
#include "../ConceptBlock/SpanTextEncoder.hpp"

namespace GRIM::Forward {

// HTTP compatibility adapter only. Realized inference uses BatchPayload and
// GoalTokenSpan metadata; no training entry is constructed at this boundary.
inline nlohmann::json inferenceInputSource(
    const std::string& prompt, const nlohmann::json& state) {
    if (prompt.empty()) throw std::invalid_argument("Inference prompt is empty");
    if (!state.is_object()) throw std::invalid_argument("Inference state must be an object");
    for (auto it = state.begin(); it != state.end(); ++it)
        if (it.key() != "goal" && it.key() != "knowns" && it.key() != "unknowns")
            throw std::invalid_argument("Unsupported inference input field: " + it.key());
    for (const auto* name : {"knowns", "unknowns"}) {
        if (!state.contains(name)) continue;
        if (!state.at(name).is_array()) throw std::invalid_argument(std::string(name) + " must be an array");
        for (const auto& entry : state.at(name)) (void)entry.get<std::string>();
    }
    if (state.contains("goal") && !state.at("goal").is_null()) {
        const auto& goal = state.at("goal");
        if (!goal.is_object()) throw std::invalid_argument("Inference goal must be an object");
        for (auto it = goal.begin(); it != goal.end(); ++it)
            if (it.key() != "target_state" && it.key() != "success_criteria" && it.key() != "constraints")
                throw std::invalid_argument("Unsupported inference goal field: " + it.key());
        if (goal.contains("target_state")) (void)goal.at("target_state").get<std::string>();
        if (goal.contains("constraints")) (void)goal.at("constraints").get<std::vector<std::string>>();
        if (goal.contains("success_criteria")) {
            if (!goal.at("success_criteria").is_array()) throw std::invalid_argument("success_criteria must be an array");
            for (const auto& entry : goal.at("success_criteria")) {
                if (!entry.is_object()) throw std::invalid_argument("Criterion must be an object");
                for (auto it = entry.begin(); it != entry.end(); ++it)
                    if (it.key() != "criterion" && it.key() != "evidence")
                        throw std::invalid_argument("Unsupported criterion field: " + it.key());
                (void)entry.at("criterion").get<std::string>();
                if (entry.contains("evidence")) (void)entry.at("evidence").get<std::string>();
            }
        }
    }
    auto source = state;
    source["prompt"] = prompt;
    return source;
}

} // namespace GRIM::Forward
