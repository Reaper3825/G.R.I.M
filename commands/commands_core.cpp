#include "commands_ai.hpp"
#include "commands_memory.hpp"
#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"
#include "commands_system.hpp"
#include "commands_aliases.hpp"   // 🔹 alias commands

#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice_speak.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "nlp.hpp"
#include "synonyms.hpp"
#include "commands_core.hpp"
#include "aliases.hpp"            // 🔹 alias resolution

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

    // 🔹 Synonym normalization
    out = normalizeWord(out);

    // 🔹 Fuzzy match
    out = fuzzyMatch(out);

    return out;
}

static std::string cleanArg(const std::string& arg) {
    std::string out;
    for (char c : arg) {
        if (std::isalnum(static_cast<unsigned char>(c)) || std::isspace(static_cast<unsigned char>(c))) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    // trim whitespace
    if (!out.empty()) {
        out.erase(0, out.find_first_not_of(" \n\r\t"));
        out.erase(out.find_last_not_of(" \n\r\t") + 1);
    }
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

        // --- Interface ---
        {"sysinfo",      cmdSystemInfo},
        {"clean",        cmdClean},
        {"help",         cmdShowHelp},
        {"reload_nlp",   cmd_reloadNLP},

        // --- Voice ---
        {"voice",        cmdVoice},
        {"voice_stream", cmdVoiceStream},
        {"test_tts",     cmd_testTTS},
        {"test_sapi",    cmd_testSAPI},
        {"tts_device",   cmd_ttsDevice},
        {"list_voice",   cmd_listVoices},

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
        std::cerr << "[DEBUG][dispatchCommand] Found handler for cmd=\"" << cmd
                  << "\" arg=\"" << arg << "\"\n";
        try {
            return it->second(arg);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR][dispatchCommand] Exception in command \"" << cmd
                      << "\": " << e.what() << "\n";
            return {
                "[Error] Exception while running command: " + cmd,
                false,
                sf::Color::Red,
                "ERR_CMD_EXCEPTION"
            };
        }
    }

    std::cerr << "[DEBUG][dispatchCommand] Unknown command: \"" << cmd << "\"\n";
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
    std::cerr << "[TRACE][handleCommand] START line=\"" << line << "\"\n";

    auto [cmdRaw, arg] = parseInput(line);
    std::cerr << "[TRACE][handleCommand] parseInput → cmdRaw=\"" << cmdRaw
              << "\" arg=\"" << arg << "\"\n";

    // Always echo user input in history (white)
    history.push("> " + line, sf::Color::White);

    // Initialize result
    CommandResult result;

    // Case 1: direct command
    if (commandMap.find(cmdRaw) != commandMap.end()) {
        std::cerr << "[TRACE][handleCommand] Direct command match: \"" << cmdRaw << "\"\n";
        result = dispatchCommand(cmdRaw, arg);
    }
    else {
        // Case 2: NLP intent
        std::cerr << "[TRACE][handleCommand] No direct match, running NLP parse...\n";

        // 🔹 Synonyms preprocessing
        std::istringstream iss(line);
        std::ostringstream oss;
        std::string token;
        while (iss >> token) {
            oss << normalizeWord(token) << " ";
        }
        std::string normalizedLine = oss.str();

        std::cerr << "[TRACE][handleCommand] Normalized line=\"" << normalizedLine << "\"\n";
        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        std::cerr << "[TRACE][handleCommand] NLP parse returned: "
                  << "name=\"" << intent.name << "\" "
                  << "matched=" << (intent.matched ? "true" : "false")
                  << " slots=" << intent.slots.size() << "\n";
        for (const auto& [k, v] : intent.slots) {
            std::cerr << "   slot[" << k << "]=\"" << v << "\"\n";
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
                arg = cleanArg(slotArg);   // 🔹 normalize punctuation, lowercase, trim
            }
        }

        std::cerr << "[TRACE][handleCommand] Final dispatch values → cmd=\"" << cmd
                  << "\" arg=\"" << arg << "\"\n";

        // Special case: open_app → resolve alias before dispatch
        if (cmd == "open_app") {
            arg = cleanArg(arg);
            std::cerr << "[DEBUG][open_app] Cleaned arg=\"" << arg << "\"\n";

            std::string resolved;
            try {
                resolved = aliases::resolve(arg);
            } catch (const std::exception& e) {
                std::cerr << "[ERROR][open_app] Exception during alias resolve: " << e.what() << "\n";
                resolved.clear();
            }

            if (resolved.empty()) {
                int bestDist = 3;
                std::string bestAlias;

                for (const auto& [alias, target] : aliases::getAll()) {
                    int dist = levenshteinDistance(normalizeWord(arg), normalizeWord(alias));
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestAlias = alias;
                        resolved = target;
                    }
                }

                if (!resolved.empty()) {
                    std::cerr << "[DEBUG][open_app] Fuzzy matched \"" << arg
                              << "\" → alias \"" << bestAlias
                              << "\" → " << resolved << "\n";
                }
            }

            if (resolved.empty()) {
                std::cerr << "[DEBUG][open_app] No alias found, using raw name: " << arg << "\n";
                resolved = arg;
            }

            result = dispatchCommand("open_app", resolved);
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

    // 🔹 Echo result back to REPL
    std::cout << finalText << std::endl;

    // ✅ Only speak real responses, never logs/traces
    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos) {
        Voice::speak(result.voice,
                     result.category.empty() ? "routine" : result.category);
    }

    std::cerr << "[TRACE][handleCommand] END\n";
}
