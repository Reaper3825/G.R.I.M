#include "commands_utils.hpp"
#include "console_history.hpp"
#include "response_manager.hpp"
#include "voice_speak.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>

// Externals
extern ConsoleHistory history;

CommandResult cmdSystemInfo([[maybe_unused]] const std::string& arg) {
    std::ostringstream oss;
    oss << "[System] OS: ";

#if defined(_WIN32)
    oss << "Windows";
#elif defined(__APPLE__)
    oss << "macOS";
#elif defined(__linux__)
    oss << "Linux";
#else
    oss << "Unknown";
#endif

    oss << "\n[System] User: " << getenv("USERNAME");
    return { oss.str(), true, sf::Color::Cyan };
}

CommandResult cmdClean([[maybe_unused]] const std::string& arg) {
    history.clear();

    history.push("[Utility] Console cleared.", sf::Color::Green);
    speak("[Utility] Console cleared.", "routine");
    return { "[Utility] Console cleared.", true, sf::Color::Green };
}

CommandResult cmdShowHelp([[maybe_unused]] const std::string& arg) {
    std::string helpText =
        "[Help] Available commands:\n"
        "- remember <key> <value>\n"
        "- recall <key>\n"
        "- forget <key>\n"
        "- ai_backend <name>\n"
        "- reload_nlp\n"
        "- pwd\n"
        "- cd <dir>\n"
        "- ls\n"
        "- mkdir <dir>\n"
        "- rm <file>\n"
        "- set_timer <seconds>\n"
        "- sysinfo\n"
        "- clean\n"
        "- help\n"
        "- voice\n"
        "- voice_stream\n";

    history.push(helpText, sf::Color::Cyan);
    speak("Listing available commands.", "routine");

    return { helpText, true, sf::Color::Cyan };
}
