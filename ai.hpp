#pragma once
#include <string>
#include <nlohmann/json.hpp>

// Global long-term memory object
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;

// Persistence functions
void loadMemory();
void saveMemory();

// AI configuration
void loadAIConfig(const std::string& path);

// Core AI call
std::string callAI(const std::string& prompt);
