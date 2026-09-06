#pragma once
#include "reasoning_state.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace GRIM {
inline nlohmann::json reasoningStateToJson(const ReasoningState& state) {
    nlohmann::json out{{"knowns", state.knowns}, {"unknowns", state.unknowns}};
    if (state.goal) {
        auto& goal = out["goal"];
        goal = {{"target_state", state.goal->target_state}, {"constraints", state.goal->constraints},
                {"success_criteria", nlohmann::json::array()}};
        for (const auto& entry : state.goal->success_criteria)
            goal["success_criteria"].push_back({{"criterion", entry.criterion}, {"evidence", entry.evidence}});
    }
    return out;
}
inline ReasoningState reasoningStateFromJson(const nlohmann::json& source) {
    if (!source.is_object()) throw std::invalid_argument("reasoning_state must be an object");
    for (auto it = source.begin(); it != source.end(); ++it)
        if (it.key() != "knowns" && it.key() != "unknowns" && it.key() != "goal")
            throw std::invalid_argument("Unsupported reasoning_state field: " + it.key());
    ReasoningState state;
    state.knowns = source.value("knowns", std::vector<std::string>{});
    state.unknowns = source.value("unknowns", std::vector<std::string>{});
    if (source.contains("goal") && !source["goal"].is_null()) {
        const auto& goal = source["goal"];
        if (!goal.is_object()) throw std::invalid_argument("reasoning_state.goal must be an object");
        for (auto it = goal.begin(); it != goal.end(); ++it)
            if (it.key() != "target_state" && it.key() != "success_criteria" && it.key() != "constraints")
                throw std::invalid_argument("Unsupported goal field: " + it.key());
        state.goal.emplace();
        state.goal->target_state = goal.value("target_state", std::string{});
        state.goal->constraints = goal.value("constraints", std::vector<std::string>{});
        if (goal.contains("success_criteria")) {
            if (!goal["success_criteria"].is_array()) throw std::invalid_argument("success_criteria must be an array");
            for (const auto& entry : goal["success_criteria"])
                state.goal->success_criteria.push_back({entry.at("criterion").get<std::string>(), entry.value("evidence", std::string{})});
        }
    }
    return state;
}
} // namespace GRIM
