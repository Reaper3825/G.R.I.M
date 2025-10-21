#pragma once
#include <string>
#include "helpers/color.hpp"
#include "commands/commands_core.hpp"

namespace ResponseManager {

    // Retrieve a response string by key or literal message
    std::string get(const std::string& keyOrMessage);

    // System/log messages (bypass NLP/commands)
    CommandResult systemMessage(const std::string& msg,
                                const Color& color = Colors::Green);
}
