#pragma once
#include <string>
#include <nlohmann/json.hpp>


nlohmann::json loadAIConfig(const std::string& path);
// Global long-term memory object
extern nlohmann::json longTermMemory;

// Persistence functions
void loadMemory();
void saveMemory();
void loadAIConfig();

// Core AI call
std::string callAI(const std::string& prompt);
