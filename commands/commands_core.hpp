#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>
#include <vector>
#include <SFML/Graphics.hpp>
#include"intent.hpp"

// Forward declarations
struct CommandResult;
struct Timer;
class ConsoleHistory;

// ------------------------------------------------------------
// CommandResult: unified return type for all commands
// ------------------------------------------------------------
struct CommandResult {
    std::string message;    // user-facing text
    bool success;           // true if command succeeded
    sf::Color color;        // console display color
    std::string errorCode;  // optional error code for ErrorManager/Logger
};

// ------------------------------------------------------------
// Function pointer type for commands
// ------------------------------------------------------------
using CommandFunc = CommandResult(*)(const std::string& arg);

// ------------------------------------------------------------
// Globals (declared here, defined in commands_core.cpp)
// ------------------------------------------------------------
extern std::unordered_map<std::string, CommandFunc> commandMap;
extern ConsoleHistory history;
extern std::vector<Timer> timers;
extern std::filesystem::path g_currentDir;
extern Intent g_lastIntent;



// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
std::pair<std::string, std::string> parseInput(const std::string& input);
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg);
void handleCommand(const std::string& line);

