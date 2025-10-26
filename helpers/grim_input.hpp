#pragma once
#include <string>
#include <vector>
#include <utility>

// Forward declarations
std::string normalizeWord(const std::string& word);

namespace GRIMInput
{
    // Split a user line into command + argument
    std::pair<std::string, std::string> parseInput(const std::string& input);

    // Split input by commas for multi-command support
    std::vector<std::string> splitCommands(const std::string& input);

    // Normalize a command: lowercase ? synonym ? fuzzy
    std::string normalizeCommand(const std::string& input);

    // Clean argument text (strip symbols, lowercase)
    std::string cleanArg(const std::string& arg);

    // Normalize an entire line for NLP intent parsing
    std::string normalizeLine(const std::string& line);
}
