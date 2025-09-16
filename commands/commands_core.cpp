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
#include "synonyms.hpp"     // 🔹 normalizeWord()
#include "commands_core.hpp"

#include <nlohmann/json.hpp>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
std::unordered_map<std::string, CommandFunc> commandMap;

// Externals
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;   // defined in nlp.cpp
extern ConsoleHistory history;
Intent g_lastIntent; // last matched intent

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------

// simple Levenshtein distance for fuzzy matching
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

    // 2. Synonym substitution
    out = normalizeWord(out);

    // 3. Fuzzy match
    out = fuzzyMatch(out);

    return out;
}

// ------------------------------------------------------------
// Command Registration
// ------------------------------------------------------------
static void initCommands() {
    if (!commandMap.empty()) return; // already initialized

    commandMap = {
        // --- Memory ---
        {"remember",     cmdRemember},
        {"recall",       cmdRecall},
        {"forget",       cmdForget},

        // --- AI / NLP ---
        {"ai_backend",   cmdAiBackend},
        {"reload_nlp",   cmdReloadNlp},
        {"grim_ai",      cmdGrimAi},   // ✅ catch-all AI queries

        // --- Filesystem ---
        {"pwd",          cmdShowPwd},
        {"cd",           cmdChangeDir},
        {"ls",           cmdListDir},
        {"mkdir",        cmdMakeDir},
        {"rm",           cmdRemoveFile},

        // --- Timers ---
        {"timer",        cmdSetTimer},

        // --- System ---
        {"sysinfo",      cmdSystemInfo},
        {"clean",        cmdClean},
        {"help",         cmdShowHelp},

        // --- Voice ---
        {"voice",        cmdVoice},
        {"voice_stream", cmdVoiceStream},

        // --- Apps / Web ---
        {"open_app",     cmdOpenApp},
        {"search_web",   cmdSearchWeb}
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
        std::cout << "[DEBUG][dispatchCommand] Found handler for cmd=\""
                  << cmd << "\" arg=\"" << arg << "\"\n";
        try {
            return it->second(arg);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR][dispatchCommand] Exception in command \""
                      << cmd << "\": " << e.what() << "\n";
            return {
                "[Error] Exception while running command: " + cmd,
                false,
                sf::Color::Red,
                "ERR_CMD_EXCEPTION"
            };
        }
    }

    std::cout << "[DEBUG][dispatchCommand] Unknown command: \"" << cmd << "\"\n";
    return {
        ErrorManager::getUserMessage("ERR_CORE_UNKNOWN_COMMAND") + ": " + cmd,
        false,
        sf::Color::Red,
        "ERR_CORE_UNKNOWN_COMMAND"
    };
}

// ------------------------------------------------------------
// handleCommand: central hub for command + NLP execution
// ------------------------------------------------------------
void handleCommand(const std::string& line) {
    // Step 1: raw input
    std::cout << "[TRACE][handleCommand] START line=\"" << line << "\"\n";

    // Step 2: parse raw into cmdRaw + arg
    auto [cmdRaw, arg] = parseInput(line);
    std::cout << "[TRACE][handleCommand] parseInput → cmdRaw=\"" << cmdRaw
              << "\" arg=\"" << arg << "\"\n";

    // Echo user input to history
    history.push("> " + line, sf::Color::White);

    // 🔹 Case 1: direct command match (skip NLP entirely)
    if (commandMap.find(cmdRaw) != commandMap.end()) {
        std::cout << "[TRACE][handleCommand] Direct command match: \"" << cmdRaw << "\"\n";
        CommandResult result = dispatchCommand(cmdRaw, arg);

        // Naturalize response
        std::string response = result.message;
        if (!result.success && result.errorCode != "ERR_NONE") {
            response = ResponseManager::get(result.message);
        } else if (!(response.size() > 0 &&
                     (response[0] == '[' || response.find('\n') != std::string::npos))) {
            response = ResponseManager::get(result.message);
        }

        Logger::logResult(result);
        history.push(response, result.color);
        if (!response.empty()) {
            speak(response, "routine");
        }
        return;
    }

    // 🔹 Case 2: NLP intent match
    std::cout << "[TRACE][handleCommand] No direct match, running NLP parse...\n";
    Intent intent = g_nlp.parse(line);
    g_lastIntent = intent;  // cache globally

    std::cout << "[TRACE][handleCommand] NLP parse returned: "
              << "name=\"" << intent.name << "\" "
              << "matched=" << (intent.matched ? "true" : "false") 
              << " slots=" << intent.slots.size() << "\n";
    for (const auto& [k, v] : intent.slots) {
        std::cout << "   slot[" << k << "]=\"" << v << "\"\n";
    }

    // Step 3: choose command
    std::string cmd = intent.matched ? intent.name : normalizeCommand(cmdRaw);
    std::cout << "[TRACE][handleCommand] selected cmd=\"" << cmd << "\"\n";

    // Step 4: extract argument from slots if matched
    if (intent.matched) {
        std::string slotArg;
        if (intent.slots.count("app") && !intent.slots.at("app").empty()) {
            slotArg = intent.slots.at("app");
        }
        else if (intent.slots.count("target") && !intent.slots.at("target").empty()) {
            slotArg = intent.slots.at("target");
        }
        else if (intent.slots.count("slot1") && !intent.slots.at("slot1").empty()) {
            slotArg = intent.slots.at("slot1");
        }
        else if (intent.slots.count("slot2") && !intent.slots.at("slot2").empty()) {
            slotArg = intent.slots.at("slot2");
        }
        else {
            for (const auto& [k, v] : intent.slots) {
                if (!v.empty()) {
                    slotArg = v;
                    break;
                }
            }
        }
        if (!slotArg.empty()) {
            arg = slotArg;
        }
    }

    std::cout << "[TRACE][handleCommand] Final dispatch values → cmd=\"" << cmd
              << "\" arg=\"" << arg << "\"\n";

    // Step 5: dispatch
    CommandResult result = dispatchCommand(cmd, arg);

    // Step 6: naturalize response
    std::string response = result.message;
    if (!result.success && result.errorCode != "ERR_NONE") {
        response = ResponseManager::get(result.message);
    } else if (!(response.size() > 0 &&
                 (response[0] == '[' || response.find('\n') != std::string::npos))) {
        response = ResponseManager::get(result.message);
    }

    Logger::logResult(result);
    history.push(response, result.color);
    if (!response.empty()) {
        speak(response, "routine");
    }

    std::cout << "[TRACE][handleCommand] END\n";
}
