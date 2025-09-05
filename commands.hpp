// commands.hpp
#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "NLP.hpp"

// Forward declarations
void handleCommand(
    const Intent& intent,
    std::string& buffer,
    std::filesystem::path& currentDir,
    std::vector<struct Timer>& timers,
    nlohmann::json& longTermMemory,
    sf::Color defaultColor = sf::Color::White
);

// Utility to normalize keys (for memory)
std::string normalizeKey(std::string key);

// Save/load long-term memory
void loadMemory();
void saveMemory();

// Context memory
void saveToMemory(const std::string& line);
std::string buildContextPrompt(const std::string& query);
