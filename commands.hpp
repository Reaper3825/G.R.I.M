#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "NLP.hpp"
#include "console_history.hpp"
#include "timer.hpp"

// Command dispatcher
// Executes based on parsed NLP intent and updates history.
bool handleCommand(const Intent& intent,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& history);
