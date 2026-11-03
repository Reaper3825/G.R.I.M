#include "EnvironmentInput.hpp"
#include "EnvironmentObservation.hpp"

#include <stdexcept>
#include <utility>

namespace GRIM::Environment {

EnvironmentInferenceInput buildEnvironmentInferenceInput(
    const ConceptBlockGoal& goal, const nlohmann::json& response,
    const std::string& task_prompt) {
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
    const auto path=state.value("location_path",std::string{});
    const auto visible=[&](const char* field){return cleanEnvironmentObservation(state.value(field,std::string{}),path);};
    const auto description=visible("description"),inventory=visible("inventory");
    auto feedback=visible("feedback");
    if(feedback==description || feedback==inventory)feedback.clear();
    std::string observation;
    const auto append = [&](const std::string& text, const char* label) {
        if (text.empty()) return;
        if (!observation.empty()) observation += "\n\n";
        observation += label;
        observation += ":\n";
        observation += text;
    };
    append(feedback, "Feedback");
    append(description, "Description");
    append(inventory, "Inventory");
    append(visible("location"), "Location");
    append(path, "Location path");
    if (input.episode_status == "active" && observation.empty())
        throw std::invalid_argument("Active environment response has no visible observation");
    input.prompt=task_prompt;

    // Strip evidence before the renderer, independently of configured span
    // policies. The original frozen Goal remains intact for worker evaluation.
    nlohmann::json visible_goal{{"target_state", goal.target_state},
        {"success_criteria", nlohmann::json::array()}, {"constraints", nlohmann::json::array()}};
    for (const auto& item : goal.success_criteria)
        visible_goal["success_criteria"].push_back({{"criterion", item.criterion}});
    for (const auto& item : goal.constraints)
        visible_goal["constraints"].push_back({{"constraint", item.constraint}});
    input.reasoning_state = {{"goal", std::move(visible_goal)},
        {"environment_context",std::move(observation)},
        {"knowns", nlohmann::json::array()}, {"unknowns", nlohmann::json::array()}};
    return input;
}

EnvironmentInferenceInput buildEnvironmentInferenceInput(
    const ConceptBlockGoal& goal, const nlohmann::json& response) {
    return buildEnvironmentInferenceInput(goal, response, {});
}

} // namespace GRIM::Environment
