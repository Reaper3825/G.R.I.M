#include "commands_ai.hpp"
#include "commands_memory.hpp"
#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"
#include "commands_system.hpp"

#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice_speak.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "nlp.hpp"          // 🔹 for g_nlp (regex-based rules)
#include "synonyms.hpp"     // 🔹 applySynonyms()
#include "commands_core.hpp"

#include <nlohmann/json.hpp>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <iostream>

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
std::unordered_map<std::string, CommandFunc> commandMap;

// Externals
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;   // loaded in nlp.cpp

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

// fuzzy match helper (simple Levenshtein-like distance)
static int levenshteinDistance(const std::string& s1, const std::string& s2) {
    const size_t m = s1.size(), n = s2.size();
    std::vector<int> prev(n + 1), curr(n + 1);

    for (size_t j = 0; j <= n; j++) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= m; i++) {
        curr[0] = static_cast<int>(i);
        for (size_t j = 1; j <= n; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            curr[j] = std::min({ prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost });
        }
        prev.swap(curr);
    }
    return prev[n];
}

static std::string fuzzyMatch(const std::string& input) {
    std::string best = input;
    int bestDist = 2; // only allow corrections within distance ≤ 2

    for (const auto& [key, _] : commandMap) {
        int dist = levenshteinDistance(input, key);
        if (dist < bestDist) {
            bestDist = dist;
            best = key;
        }
    }
    return best;
}

static std::string normalizeCommand(const std::string& input) {
    // 1. Lowercase
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    // 2. Synonym substitution (phrases/words → canonical command)
    out = applySynonyms(out);

    // 3. Fuzzy match (typos → canonical command)
    out = fuzzyMatch(out);

    // 4. NLP regex parse (maps flexible phrasing → canonical command)
    auto intent = g_nlp.parse(out);
    if (!intent.intent.empty()) {
        out = intent.intent;  // must equal a commandMap key
    }

    return out;
}

// ------------------------------------------------------------
// Command Registration
// ------------------------------------------------------------
static void initCommands() {
    if (!commandMap.empty()) return; // already initialized

    commandMap = {
        {"remember",     cmdRemember},
        {"recall",       cmdRecall},
        {"forget",       cmdForget},
        {"ai_backend",   cmdAiBackend},
        {"reload_nlp",   cmdReloadNlp},
        {"pwd",          cmdShowPwd},
        {"cd",           cmdChangeDir},
        {"ls",           cmdListDir},
        {"mkdir",        cmdMakeDir},
        {"rm",           cmdRemoveFile},
        {"timer",        cmdSetTimer},
        {"sysinfo",      cmdSystemInfo},
        {"clean",        cmdClean},
        {"help",         cmdShowHelp},
        {"voice",        cmdVoice},
        {"voice_stream", cmdVoiceStream}
    };
}

// ------------------------------------------------------------
// Core Dispatch
// ------------------------------------------------------------
std::pair<std::string, std::string> parseInput(const std::string& input) {
    auto pos = input.find(' ');
    if (pos == std::string::npos) {
        return {input, ""};
    }
    return {input.substr(0, pos), input.substr(pos + 1)};
}

CommandResult dispatchCommand(const std::string& cmd, const std::string& arg) {
    initCommands();

    auto it = commandMap.find(cmd);
    if (it != commandMap.end()) {
        return it->second(arg);
    }

    // Unknown command
    return {
        ErrorManager::getUserMessage("ERR_CORE_UNKNOWN_COMMAND") + ": " + cmd,
        false,
        sf::Color::Red,
        "ERR_CORE_UNKNOWN_COMMAND"
    };
}

void handleCommand(const std::string& line) {
    auto [cmdRaw, arg] = parseInput(line);

    // 🔹 Full NLP pipeline normalization
    std::string cmd = normalizeCommand(cmdRaw);

    // 🔹 Dispatch
    CommandResult result = dispatchCommand(cmd, arg);

    // 🔹 Log result
    Logger::logResult(result);

    // 🔹 Push to history and speak
    history.push(ResponseManager::get(result.message), result.color);
    if (result.success) {
        speak(result.message, "routine");
    }
}
