#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <deque>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "NLP.hpp"
#include "console_history.hpp"
#include "timer.hpp"


// ---------------- Global Memory ----------------
extern std::deque<std::string> contextMemory;
extern const size_t kMaxContext;
extern nlohmann::json longTermMemory;

// ---------------- Memory Helpers ----------------
std::string normalizeKey(std::string key);
void loadMemory();
void saveMemory();
void saveToMemory(const std::string& line);
std::string buildContextPrompt(const std::string& query);

// ---------------- Command Handler ----------------
// Called by main.cpp after NLP parses user input
void handleCommand(
    const Intent& intent,
    std::string& buffer,
    std::filesystem::path& currentDir,
    std::vector<Timer>& timers,
    nlohmann::json& longTermMemory,
    NLP& nlp,
    ConsoleHistory& history
);

