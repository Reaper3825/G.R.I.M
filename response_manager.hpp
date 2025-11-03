#pragma once
#include <string>
#include <unordered_map>
#include "helpers/color.hpp"
#include "commands/commands_core.hpp"

namespace ResponseManager {

    // Retrieve a response string by key or literal message
    std::string get(const std::string& keyOrMessage);

    // Get a response with parameter substitution
    std::string getWithParams(const std::string& key, 
                             const std::unordered_map<std::string, std::string>& params);

    // System/log messages (bypass NLP/commands)
    CommandResult systemMessage(const std::string& msg,
                                const Color& color = Colors::Green);

    // Get contextual greeting based on time of day
    std::string getGreeting();

    // Clear response history (for testing or reset)
    void clearHistory();
}
