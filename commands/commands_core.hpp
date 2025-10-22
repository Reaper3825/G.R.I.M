#pragma once
#include <string>
#include <unordered_map>
#include <functional>
#include <optional>
#include <filesystem>
#include "helpers/color.hpp"
#include "intent.hpp"

// ====================================================
// Command result structure
// ====================================================
struct CommandResult {
    bool success = true;
    std::string message;
    std::string errorCode;
    std::string category;
    std::string voice;
    Color color = Colors::Default;
};

// ====================================================
// Command function type
// ====================================================
using CommandFunc = std::function<CommandResult(const std::string&)>;

// ====================================================
// Globals
// ====================================================
extern std::unordered_map<std::string, CommandFunc> commandMap;

// Correct type restored — matches old GRIM core
extern std::filesystem::path g_currentDir;

// Last NLP intent (for "nevermind" command)
extern Intent g_lastIntent;

// ====================================================
// Core functions
// ====================================================

// Parse user input into (command, argument)
std::pair<std::string, std::string> parseInput(const std::string& input);

// Dispatch a specific command by name
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg);

// Handle command from raw input line (main entry point)
void handleCommand(const std::string& line);

// Ensure that built-in / core plugins are registered
void ensureCorePluginsRegistered();

// ====================================================
// NLP normalization helpers
// ====================================================
std::string normalizeWord(const std::string& word);
std::string normalizeLine(const std::string& line);

// ✅ Check if feedback is currently pending
bool hasPendingFeedback();

// ✅ NEW: Set whether this command came from voice (for feedback control)
void setVoiceCommand(bool isVoice);
