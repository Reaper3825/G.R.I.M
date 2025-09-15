#pragma once
#include <string>
#include <SFML/Graphics/Color.hpp>

// Represents the result of a command execution
struct CommandResult {
    std::string message;   // What to show in console
    bool success;          // Command succeeded?
    sf::Color color;       // Console text color
};

// Function prototypes for commands
CommandResult cmdClean(const std::string& arg);
CommandResult cmdShowHelp(const std::string& arg);

// You’ll also want to declare the rest here, e.g.:
CommandResult cmdAiBackend(const std::string& arg);
CommandResult cmdReloadNlp(const std::string& arg);
CommandResult cmdSetTimer(const std::string& arg);
CommandResult cmdSystemInfo(const std::string& arg);
CommandResult cmdVoice(const std::string& arg);
CommandResult cmdVoiceStream(const std::string& arg);
CommandResult cmdRemember(const std::string& arg);
CommandResult cmdRecall(const std::string& arg);
CommandResult cmdForget(const std::string& arg);
