#pragma once
#include <string>
#include <filesystem>   // needed for std::filesystem::path

namespace fs = std::filesystem;   // bring in shorthand "fs"

// forward declarations
bool loadNlpRules(const std::string& path);
std::string mapTextToCommand(const std::string& text);

// declare the NLP parser correctly
std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir);
