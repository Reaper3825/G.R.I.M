#include "commands_ai.hpp"
#include "commands_memory.hpp"
#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"
#include "commands_system.hpp"
#include "commands_aliases.hpp"
#include "response_manager.hpp"
#include "console_history.hpp"
#include "voice/voice_speak.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "nlp/nlp.hpp"
#include "synonyms.hpp"
#include "commands_core.hpp"
#include "aliases.hpp"
#include "ai/personality_manager.hpp"
#include "ai/proactive_dialogue.hpp"
#include "logger.hpp"
#include <sstream>
#include "memory/context_manager.hpp"

using Voice::speak;

// ====================================================
// Globals
// ====================================================
std::unordered_map<std::string, CommandFunc> commandMap;

// Externals
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
extern ConsoleHistory history;
Intent g_lastIntent;

// ====================================================
// Helpers
// ====================================================
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
    int bestDist = 2;

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

    out = normalizeWord(out);
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
    if (!out.empty()) {
        out.erase(0, out.find_first_not_of(" \n\r\t"));
        out.erase(out.find_last_not_of(" \n\r\t") + 1);
    }
    return out;
}

// ====================================================
// Command Registration
// ====================================================
static void initCommands() {
    if (!commandMap.empty()) return;

    commandMap = {
        {"remember",     cmdRemember},
        {"recall",       cmdRecall},
        {"forget",       cmdForget},
        {"ai_backend",   cmdAiBackend},
        {"reload_nlp",   cmdReloadNlp},
        {"grim_ai",      cmdGrimAi},
        {"pwd",          cmdShowPwd},
        {"cd",           cmdChangeDir},
        {"ls",           cmdListDir},
        {"mkdir",        cmdMakeDir},
        {"rm",           cmdRemoveFile},
        {"timer",        cmdSetTimer},
        {"sysinfo",      cmdSystemInfo},
        {"clean",        cmdClean},
        {"help",         cmdShowHelp},
        {"reload_nlp",   cmd_reloadNLP},
        {"voice",        cmdVoice},
        {"voice_stream", cmdVoiceStream},
        {"test_tts",     cmd_testTTS},
        {"test_sapi",    cmd_testSAPI},
        {"tts_device",   cmd_ttsDevice},
        {"list_voice",   cmd_listVoices},
        {"open_app",     cmdOpenApp},
        {"search_web",   cmdSearchWeb},
        {"alias list",    cmdAliasList},
        {"alias info",    cmdAliasInfo},
        {"alias refresh", cmdAliasRefresh}
    };
}

// ====================================================
// Core Dispatch
// ====================================================
std::pair<std::string, std::string> parseInput(const std::string& input) {
    auto pos = input.find(' ');
    if (pos == std::string::npos) return {input, ""};
    return {input.substr(0, pos), input.substr(pos + 1)};
}

CommandResult dispatchCommand(const std::string& cmd, const std::string& arg) {
    initCommands();

    auto it = commandMap.find(cmd);
    if (it != commandMap.end()) {
        LOG_DEBUG("Dispatch", "Running command \"" + cmd + "\" arg=\"" + arg + "\"");
        try {
            return it->second(arg);
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
            return {"[Error] Exception while running command: " + cmd, false, sf::Color::Red, "ERR_CMD_EXCEPTION"};
        }
    }

    LOG_DEBUG("Dispatch", "Unknown command: \"" + cmd + "\"");
    return {ErrorManager::getUserMessage("ERR_CORE_UNKNOWN_COMMAND") + ": " + cmd, false, sf::Color::Red, "ERR_CORE_UNKNOWN_COMMAND"};
}

// ====================================================
// handleCommand: central hub for command + NLP execution
// ====================================================
void handleCommand(const std::string& line) {
    LOG_TRACE("HandleCommand", "START line=\"" + line + "\"");

    auto [cmdRaw, arg] = parseInput(line);
    LOG_TRACE("HandleCommand", "Parsed → cmdRaw=\"" + cmdRaw + "\" arg=\"" + arg + "\"");

    history.push("> " + line, sf::Color::White);
    CommandResult result;

    if (commandMap.find(cmdRaw) != commandMap.end()) {
        LOG_TRACE("HandleCommand", "Direct command match: \"" + cmdRaw + "\"");
        result = dispatchCommand(cmdRaw, arg);
    } else {
        LOG_TRACE("HandleCommand", "No direct match, running NLP parse...");

        std::istringstream iss(line);
        std::ostringstream oss;
        std::string token;
        while (iss >> token) oss << normalizeWord(token) << " ";
        std::string normalizedLine = oss.str();

        LOG_TRACE("HandleCommand", "Normalized line=\"" + normalizedLine + "\"");
        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        LOG_TRACE("NLP", "Intent name=\"" + intent.name + "\" matched=" +
            (intent.matched ? "true" : "false") + " slots=" + std::to_string(intent.slots.size()));

        for (const auto& [k, v] : intent.slots)
            LOG_TRACE("NLP", "Slot[" + k + "]=\"" + v + "\"");

        std::string cmd = intent.matched ? intent.name : normalizeCommand(cmdRaw);

        if (intent.matched) {
            std::string slotArg;
            if (intent.slots.count("app") && !intent.slots.at("app").empty())
                slotArg = intent.slots.at("app");
            else if (intent.slots.count("target") && !intent.slots.at("target").empty())
                slotArg = intent.slots.at("target");
            else {
                for (const auto& [k, v] : intent.slots)
                    if (!v.empty()) { slotArg = v; break; }
            }
            if (!slotArg.empty()) arg = cleanArg(slotArg);
        }

        LOG_TRACE("HandleCommand", "Final dispatch values → cmd=\"" + cmd + "\" arg=\"" + arg + "\"");

        if (cmd == "open_app") {
            arg = cleanArg(arg);
            LOG_DEBUG("OpenApp", "Cleaned arg=\"" + arg + "\"");
            std::string resolved;
            try {
                resolved = aliases::resolve(arg);
            } catch (const std::exception& e) {
                LOG_ERROR("OpenApp", "Alias resolution failed: " + std::string(e.what()));
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
                if (!resolved.empty())
                    LOG_DEBUG("OpenApp", "Fuzzy matched \"" + arg + "\" → alias \"" + bestAlias + "\" → " + resolved);
            }

            if (resolved.empty()) {
                LOG_DEBUG("OpenApp", "No alias found, using raw name: " + arg);
                resolved = arg;
            }

            result = dispatchCommand("open_app", resolved);
        }
    }

    if (result.message.empty()) {
        result.message = "[no response configured]";
        result.success = false;
        if (result.errorCode.empty()) result.errorCode = "ERR_NONE";
    }

    std::string finalText = ResponseManager::get(result.message);
    Logger::logResult(result);
    history.push(finalText, result.color);

    LOG_DEBUG("HandleCommand", "Output → " + finalText);
    std::cout << finalText << std::endl;

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos)
        Voice::speak(result.voice, result.category.empty() ? "routine" : result.category);

    // 🔹 Update GRIM's emotional state
    GRIM::ContextManager::recordUsage(cmdRaw);
    GRIM::PersonalityManager::updateAfterCommand(result.success);
    GRIM::DialogueProactive::checkAfterCommand(line, result);
    

    LOG_TRACE("HandleCommand", "END");
}
