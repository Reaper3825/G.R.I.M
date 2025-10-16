#pragma once
#include <string>
#include <utility>

namespace GRIMInput
{
    std::pair<std::string, std::string> parseInput(const std::string& input);
    std::string normalizeCommand(const std::string& input);
    std::string cleanArg(const std::string& arg);
    std::string normalizeLine(const std::string& line);
}
