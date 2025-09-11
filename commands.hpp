#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "nlp.hpp"
#include "console_history.hpp"
#include "timer.hpp"

/// Dispatches a parsed NLP intent into the correct action.
/// Updates console history, modifies timers, filesystem state, or memory.
/// Returns true if the command was handled (even if with an error message).
///
/// @param intent          The parsed NLP intent (name + slots).
/// @param buffer          The live input buffer (modified for some commands).
/// @param currentDir      Reference to the current working directory.
/// @param timers          Active timers vector (for alarms).
/// @param longTermMemory  Persistent key/value memory (saved to JSON).
/// @param nlp             NLP engine (may reload rules).
/// @param history         Console history log for output.
bool handleCommand(const Intent& intent,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& history);
