#pragma once
#include <string>
#include <nlohmann/json.hpp>

// Global long-term memory object
extern nlohmann::json longTermMemory;

// Persistence functions
void loadMemory();
void saveMemory();

// Core AI call
std::string callAI(const std::string& prompt);
