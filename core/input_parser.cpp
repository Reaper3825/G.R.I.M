#include "input_parser.hpp"
#include "commands_core.hpp"     // for CommandResult / CommandFunc / commandMap
#include "synonyms.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cctype>
#include <vector>
#include <sstream>
#include <unordered_map>

namespace GRIMInput
{
    // ====================================================
    // Helper: Levenshtein distance (edit distance)
    // ====================================================
    static int levenshteinDistance(const std::string& s1, const std::string& s2)
    {
        const size_t m = s1.size(), n = s2.size();
        std::vector<int> prev(n + 1), curr(n + 1);
        for (size_t j = 0; j <= n; ++j)
            prev[j] = static_cast<int>(j);

        for (size_t i = 1; i <= m; ++i)
        {
            curr[0] = static_cast<int>(i);
            for (size_t j = 1; j <= n; ++j)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                curr[j] = std::min({ prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost });
            }
            prev.swap(curr);
        }
        return prev[n];
    }

    // ====================================================
    // Helper: Fuzzy match to known commands
    // ====================================================
    static std::string fuzzyMatch(const std::string& input)
    {
        std::string best = input;
        int bestDist = 2; // tolerate minor typos

        for (const auto& [key, _] : commandMap)
        {
            int dist = levenshteinDistance(input, key);
            if (dist < bestDist)
            {
                bestDist = dist;
                best = key;
            }
        }
        return best;
    }

    // ====================================================
    // Split a user line into command + argument
    // ====================================================
    std::pair<std::string, std::string> parseInput(const std::string& input)
    {
        std::string line = input;
        if (line.empty())
            return { "", "" };

        // trim both ends
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        auto pos = line.find(' ');
        if (pos == std::string::npos)
            return { line, "" };
        return { line.substr(0, pos), line.substr(pos + 1) };
    }

    // ====================================================
    // Normalize a command: lowercase → synonym → fuzzy
    // ====================================================
    std::string normalizeCommand(const std::string& input)
    {
        if (input.empty())
            return input;

        std::string out = input;
        std::transform(out.begin(), out.end(), out.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        out = normalizeWord(out);   // synonym normalization
        out = fuzzyMatch(out);      // fuzzy correction
        return out;
    }

    // ====================================================
    // Clean argument text (strip symbols, lowercase)
    // ====================================================
    std::string cleanArg(const std::string& arg)
    {
        if (arg.empty())
            return "";

        std::string out;
        out.reserve(arg.size());

        for (char c : arg)
        {
            if (std::isalnum(static_cast<unsigned char>(c)) ||
                std::isspace(static_cast<unsigned char>(c)))
            {
                out.push_back(static_cast<char>(std::tolower(c)));
            }
        }

        // trim
        if (!out.empty())
        {
            out.erase(0, out.find_first_not_of(" \n\r\t"));
            out.erase(out.find_last_not_of(" \n\r\t") + 1);
        }
        return out;
    }

    // ====================================================
    // Normalize an entire line for NLP intent parsing
    // ====================================================
    std::string normalizeLine(const std::string& line)
    {
        std::istringstream iss(line);
        std::ostringstream oss;
        std::string token;

        while (iss >> token)
            oss << normalizeWord(token) << ' ';

        std::string result = oss.str();
        if (!result.empty() && result.back() == ' ')
            result.pop_back();
        return result;
    }
}
