#pragma once
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Try to translate a natural language request into a command.
// Returns empty string if no mapping found.
std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir);
