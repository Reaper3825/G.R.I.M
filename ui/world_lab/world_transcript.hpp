#pragma once
#include <regex>
#include <sstream>
#include <string>

namespace GRIM::WorldLab {
// Presentation-only cleanup: keep the original TextWorld state for inspection.
inline std::string cleanTranscript(const std::string& source) {
    static const std::regex interpreterLine(
        R"(^[ \t]*(>[ \t]*)?(-=[^\r\n]*=-[ \t]*-?[0-9]+[ \t]*/[ \t]*[0-9]+[ \t]*)?$)");
    std::istringstream input(source);
    std::string line, result;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Empty lines are prose spacing; only remove nonempty UI artifacts.
        if (line.find_first_not_of(" \t") != std::string::npos &&
            std::regex_match(line, interpreterLine)) continue;
        if (!result.empty()) result += '\n';
        result += line;
    }
    const auto end = result.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) return {};
    result.erase(end + 1);
    return result;
}
}
