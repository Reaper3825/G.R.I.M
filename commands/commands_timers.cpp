#include "commands_timers.hpp"
#include "console_history.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
#include <vector>

// Externals
extern std::vector<Timer> timers;
extern ConsoleHistory history;

CommandResult cmdSetTimer(const std::string& arg) {
    if (arg.empty()) return { "[Timer] Usage: set_timer <seconds>", false, sf::Color::Red };

    try {
        float seconds = std::stof(arg);
        Timer t(seconds);
        timers.push_back(t);

        return { "[Timer] Timer set for " + std::to_string(seconds) + " seconds.", true, sf::Color::Green };
    } catch (const std::exception&) {
        return { "[Timer] Invalid number: " + arg, false, sf::Color::Red };
    }
}
