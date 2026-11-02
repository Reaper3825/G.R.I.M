#include "EnvironmentInput.hpp"

#include <stdexcept>
#include <utility>

namespace GRIM::Environment {

EnvironmentInferenceInput buildEnvironmentInferenceInput(
    const ConceptBlockGoal& goal, const nlohmann::json& response) {
    if (!response.value("ok", false))
        throw std::invalid_argument("Environment input requires a successful worker response");
    const auto& state = response.at("result").at("state");
    EnvironmentInferenceInput input;
    input.episode_status = state.at("episode").at("status").get<std::string>();
    if (input.episode_status != "active" && input.episode_status != "succeeded" &&
        input.episode_status != "failed" && input.episode_status != "stopped")
        throw std::invalid_argument("Environment input has an invalid episode status");
    if (state.at("done").get<bool>() != (input.episode_status != "active"))
        throw std::invalid_argument("Environment input has inconsistent episode termination");

    // Host catalog/details/UI services already place their visible result in
    // feedback. Do not serialize their structured response envelopes a second
    // time or infer observations from the authoritative world document/facts.
    const auto append = [&](const char* field, const char* label) {
        if (!state.contains(field)) return;
        const auto text = state.at(field).get<std::string>();
        if (text.empty()) return;
        if (!input.prompt.empty()) input.prompt += "\n\n";
        input.prompt += label;
        input.prompt += ":\n";
        input.prompt += text;
    };
    append("feedback", "Feedback");
    append("description", "Description");
    append("inventory", "Inventory");
    append("location", "Location");
    append("location_path", "Location path");
    if (input.episode_status == "active" && input.prompt.empty())
        throw std::invalid_argument("Active environment response has no visible observation");

    // Strip evidence before the renderer, independently of configured span
    // policies. The original frozen Goal remains intact for worker evaluation.
    nlohmann::json visible_goal{{"target_state", goal.target_state},
        {"success_criteria", nlohmann::json::array()}, {"constraints", nlohmann::json::array()}};
    for (const auto& item : goal.success_criteria)
        visible_goal["success_criteria"].push_back({{"criterion", item.criterion}});
    for (const auto& item : goal.constraints)
        visible_goal["constraints"].push_back({{"constraint", item.constraint}});
    input.reasoning_state = {{"goal", std::move(visible_goal)},
        {"knowns", nlohmann::json::array()}, {"unknowns", nlohmann::json::array()}};
    return input;
}

} // namespace GRIM::Environment
