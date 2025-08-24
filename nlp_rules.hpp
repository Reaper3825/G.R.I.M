#pragma once
#include <string>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

struct NlpRule {
    std::string id;
    std::vector<std::string> keywords;  // optional
    std::string regex;                  // optional
    std::string command;                // required
    // simple defaults map: key -> value (as string)
    std::vector<std::pair<std::string,std::string>> defaults;
};

// Load rules from JSON file. Returns true on success.
bool loadNlpRules(const fs::path& path);

// Try to map free-form text to a concrete command using loaded rules.
// Returns empty string if no rule matched.
std::string mapTextToCommand(const std::string& inputLowered);
