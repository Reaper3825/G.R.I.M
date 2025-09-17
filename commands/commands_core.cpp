#include "commands_ai.hpp"
#include "commands_memory.hpp"
#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"
#include "commands_system.hpp"
#include "commands_aliases.hpp"   // 🔹 new: alias commands

#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice_speak.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "nlp.hpp"
#include "synonyms.hpp"
#include "commands_core.hpp"
#include "aliases.hpp"            // 🔹 new: alias resolution

#include <nlohmann/json.hpp>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>
#include <sstream>

using Voice::speak;


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
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    // 🔹 Synonym normalization (maps e.g. "launch" -> "open_app")
    out = normalizeWord(out);

    // 🔹 Fuzzy match against known command keys
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
        {"search_web",   cmdSearchWeb},

        // --- Aliases ---
        {"alias list",    cmdAliasList},
        {"alias info",    cmdAliasInfo},
        {"alias refresh", cmdAliasRefresh}
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
    std::cout << "[TRACE][handleCommand] START line=\"" << line << "\"\n";

    auto [cmdRaw, arg] = parseInput(line);
    std::cout << "[TRACE][handleCommand] parseInput → cmdRaw=\"" << cmdRaw
              << "\" arg=\"" << arg << "\"\n";

    // Always echo user input in history (white)
    history.push("> " + line, sf::Color::White);

    // Initialize result
    CommandResult result;

    // Case 1: direct command
    if (commandMap.find(cmdRaw) != commandMap.end()) {
        std::cout << "[TRACE][handleCommand] Direct command match: \"" << cmdRaw << "\"\n";
        result = dispatchCommand(cmdRaw, arg);
    }
    else {
        // Case 2: NLP intent
        std::cout << "[TRACE][handleCommand] No direct match, running NLP parse...\n";

        // 🔹 Synonyms preprocessing
        std::istringstream iss(line);
        std::ostringstream oss;
        std::string token;
        while (iss >> token) {
            oss << normalizeWord(token) << " ";
        }
        std::string normalizedLine = oss.str();

        std::cout << "[TRACE][handleCommand] Normalized line=\"" << normalizedLine << "\"\n";
        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        std::cout << "[TRACE][handleCommand] NLP parse returned: "
                  << "name=\"" << intent.name << "\" "
                  << "matched=" << (intent.matched ? "true" : "false")
                  << " slots=" << intent.slots.size() << "\n";
        for (const auto& [k, v] : intent.slots) {
            std::cout << "   slot[" << k << "]=\"" << v << "\"\n";
        }

        std::string cmd = intent.matched ? intent.name : normalizeCommand(cmdRaw);

        // Fill arg from slots if present
        if (intent.matched) {
            std::string slotArg;
            if (intent.slots.count("app") && !intent.slots.at("app").empty()) {
                slotArg = intent.slots.at("app");
            } else if (intent.slots.count("target") && !intent.slots.at("target").empty()) {
                slotArg = intent.slots.at("target");
            } else {
                for (const auto& [k, v] : intent.slots) {
                    if (!v.empty()) { slotArg = v; break; }
                }
            }
            if (!slotArg.empty()) {
                arg = slotArg;
            }
        }

        std::cout << "[TRACE][handleCommand] Final dispatch values → cmd=\"" << cmd
                  << "\" arg=\"" << arg << "\"\n";

        // Special case: open_app → resolve alias before dispatch
        if (cmd == "open_app") {
            std::string resolved = aliases::resolve(arg);
            if (resolved.empty()) {
                result = {
                    ErrorManager::getUserMessage("ERR_ALIAS_NOT_FOUND") + ": " + arg,
                    false,
                    sf::Color::Red,
                    "ERR_ALIAS_NOT_FOUND"
                };
            } else {
                std::cout << "[DEBUG][open_app] alias \"" << arg << "\" → " << resolved << "\n";
                arg = resolved;
                result = dispatchCommand("open_app", arg);
            }
        } else {
            result = dispatchCommand(cmd, arg);
        }
    }

    // 🔹 Unified output block
    if (result.message.empty()) {
        result.message = "[no response configured]";
        result.success = false;
        if (result.errorCode.empty()) result.errorCode = "ERR_NONE";
    }

    std::string finalText = ResponseManager::get(result.message);

    Logger::logResult(result);
    history.push(finalText, result.color);

    if (!result.voice.empty()) {
        speak(result.voice, result.category.empty() ? "routine" : result.category);
    }

    std::cout << "[TRACE][handleCommand] END\n";
}
