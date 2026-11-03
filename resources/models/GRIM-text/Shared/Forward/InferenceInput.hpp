#pragma once
#include "../ConceptBlock/SpanTextEncoder.hpp"
#include "../Goal/GoalContract.hpp"

namespace GRIM::Forward {

// HTTP compatibility adapter only. Realized inference uses BatchPayload and
// GoalTokenSpan metadata; no training entry is constructed at this boundary.
inline nlohmann::json inferenceInputSource(
    const std::string& prompt, const nlohmann::json& state) {
    if (!state.is_object()) throw std::invalid_argument("Inference state must be an object");
    if (prompt.empty() && state.value("environment_context",std::string{}).empty())
        throw std::invalid_argument("Inference prompt and environment context are empty");
    for (auto it = state.begin(); it != state.end(); ++it)
        if (it.key() != "goal" && it.key() != "knowns" && it.key() != "unknowns" && it.key() != "environment_context")
            throw std::invalid_argument("Unsupported inference input field: " + it.key());
    for (const auto* name : {"knowns", "unknowns"}) {
        if (!state.contains(name)) continue;
        if (!state.at(name).is_array()) throw std::invalid_argument(std::string(name) + " must be an array");
        for (const auto& entry : state.at(name)) (void)entry.get<std::string>();
    }
    if (state.contains("goal") && !state.at("goal").is_null()) {
        (void)goalFromJson(state.at("goal"));
    }
    auto source = state;
    if(state.contains("environment_context"))(void)state.at("environment_context").get<std::string>();
    source["prompt"] = prompt;
    return source;
}

} // namespace GRIM::Forward
