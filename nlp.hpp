#pragma once
#include <string>

bool loadNlpRules(const std::string& path);   // <- no noexcept to avoid mismatch
std::string mapTextToCommand(const std::string& lowered);
std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir);