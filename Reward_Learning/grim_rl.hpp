#pragma once

#include <optional>
#include <string>
#include <nlohmann/json.hpp>

struct CommandResult;

namespace GRIM::RewardLearning {

// Ask the active RL backend for a better command before dispatching.
std::optional<std::string> suggestPreDispatchCommand(
    const std::string& userInput,
    const std::string& normalizedCommand,
    const std::string& argument,
    const nlohmann::json& context,
    const std::string& mood);

// Let RL propose a correction when we do not recognize a command.
std::optional<std::string> suggestForUnknownCommand(
    const std::string& normalizedCommand,
    const std::string& argument,
    const nlohmann::json& context);

// Forward execution feedback to the RL backend with consistent logging.
void sendCommandFeedback(
    const CommandResult& result,
    const std::string& executedCommand,
    float execTime,
    float sentimentScore,
    float diversityFactor,
    const std::string& mood);

} // namespace GRIM::RewardLearning
