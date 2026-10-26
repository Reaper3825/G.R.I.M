#pragma once
#include "../ConceptBlock/SpanTextEncoder.hpp"
#include "../Goal/GoalContract.hpp"

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
        (void)goalFromJson(state.at("goal"));
    }
    auto source = state;
    source["prompt"] = prompt;
    return source;
}

} // namespace GRIM::Forward
