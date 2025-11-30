#include "Reward_Learning/grim_rl.hpp"

#include "ai/ai_rl.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"

namespace GRIM::RewardLearning {

namespace {

std::optional<std::string> extractSuggestion(const nlohmann::json& response)
{
    if (!response.contains("suggested_command"))
        return std::nullopt;

    try {
        auto suggestion = response["suggested_command"].get<std::string>();
        if (!suggestion.empty())
            return suggestion;
    } catch (const std::exception& e) {
        LOG_ERROR("RewardLearning", std::string("Invalid RL response payload: ") + e.what());
    }
    return std::nullopt;
}

bool isRegisteredCommand(const std::string& cmd)
{
    return commandMap.find(cmd) != commandMap.end();
}

} // namespace

std::optional<std::string> suggestPreDispatchCommand(
    const std::string& userInput,
    const std::string& normalizedCommand,
    const std::string& argument,
    const nlohmann::json& context,
    const std::string& mood)
{
    try {
        nlohmann::json obs = {
            {"input", userInput},
            {"command_raw", normalizedCommand},
            {"argument", argument},
            {"context", context},
            {"mood", mood}
        };

        auto response = GRIM::RL::getAction(obs);
        auto suggestion = extractSuggestion(response);
        if (!suggestion)
            return std::nullopt;

        if (isRegisteredCommand(*suggestion)) {
            LOG_DEBUG("RewardLearning", "Pre-dispatch RL suggested: " + *suggestion);
            return suggestion;
        }

        LOG_DEBUG("RewardLearning", "RL suggested unknown command: " + *suggestion);
    } catch (const std::exception& e) {
        LOG_ERROR("RewardLearning", std::string("Pre-dispatch RL failed: ") + e.what());
    }
    return std::nullopt;
}

std::optional<std::string> suggestForUnknownCommand(
    const std::string& normalizedCommand,
    const std::string& argument,
    const nlohmann::json& context)
{
    try {
        nlohmann::json obs = {
            {"type", "unknown_command"},
            {"input", argument.empty() ? normalizedCommand : normalizedCommand + " " + argument},
            {"context", context}
        };

        auto response = GRIM::RL::getAction(obs);
        auto suggestion = extractSuggestion(response);
        if (!suggestion)
            return std::nullopt;

        if (isRegisteredCommand(*suggestion)) {
            LOG_DEBUG("RewardLearning", "RL fallback suggested: " + *suggestion);
            return suggestion;
        }

        LOG_DEBUG("RewardLearning", "RL fallback command not registered: " + *suggestion);
    } catch (const std::exception& e) {
        LOG_ERROR("RewardLearning", std::string("Unknown-command RL failed: ") + e.what());
    }
    return std::nullopt;
}

void sendCommandFeedback(
    const CommandResult& result,
    const std::string& executedCommand,
    float execTime,
    float sentimentScore,
    float diversityFactor,
    const std::string& mood)
{
    try {
        GRIM::RL::processCommandResult(
            result,
            executedCommand,
            execTime,
            sentimentScore,
            diversityFactor,
            mood);
        LOG_DEBUG("RewardLearning", "Queued RL feedback for \"" + executedCommand + "\"");
    } catch (const std::exception& e) {
        LOG_ERROR("RewardLearning", std::string("Failed to queue RL feedback: ") + e.what());
    }
}

} // namespace GRIM::RewardLearning
