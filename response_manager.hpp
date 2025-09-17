#pragma once
#include <string>
#include <SFML/Graphics/Color.hpp>
#include "commands/commands_core.hpp"

namespace ResponseManager {
    std::string get(const std::string& keyOrMessage);

    // System/log messages (bypass NLP/commands)
    CommandResult systemMessage(const std::string& msg,
                                const sf::Color& color = sf::Color::Green);
}
