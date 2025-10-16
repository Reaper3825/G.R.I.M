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
#include "memory/context_manager.hpp"
#include "ai/ai_rl.hpp"
#include "memory/memory_manager.hpp"
#include "memory/memory_storage.hpp"
#include "memory/memory_router.hpp"
#include "input_parser.hpp"  // <--- NEW include

using Voice::speak;

// ====================================================
// Globals
// ====================================================
std::unordered_map<std::string, CommandFunc> commandMap;
static std::optional<std::string> g_pendingClarifyCmd;

// Externals
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
extern ConsoleHistory history;
extern GRIM::MemoryStorage g_memoryStorage;

Intent g_lastIntent;

// ====================================================
// Global learned command map + handler
// ====================================================
static std::unordered_map<std::string, std::string> g_learnedCommandMap;

static CommandResult handleLearnedCommand(const std::string& arg)
{
    for (auto& pair : g_learnedCommandMap)
    {
        const std::string& phrase = pair.first;
        const std::string& action = pair.second;

        if (arg == phrase || arg.find(phrase) != std::string::npos)
        {
            LOG_DEBUG("LearnedCmd", "Executing learned command \"" + phrase + "\" → \"" + action + "\"");
            return dispatchCommand(action, "");
        }
    }

    return {"[Error] Unknown learned command route.", false, sf::Color::Red};
}

// ====================================================
// InitCommands
// ====================================================
static void initCommands()
{
    if (!commandMap.empty())
        return;

    // === 1. Built-in commands ===
    commandMap = {
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
        {"alias refresh", cmdAliasRefresh},
        {"nevermind",     cmdNevermind}
    };

    // === 2. Merge persistent learned commands ===
    try
    {
        g_learnedCommandMap.clear();
        auto learnedList = g_memoryStorage.search("", 1000);
        int count = 0;

        for (const auto& obj : learnedList)
        {
            if (obj.type != GRIM::TypeTag::LearnedCommand)
                continue;

            const std::string& phrase = obj.raw;
            const std::string& action = obj.normalized;

            if (commandMap.find(phrase) != commandMap.end())
                continue;

            g_learnedCommandMap[phrase] = action;
            commandMap[phrase] = handleLearnedCommand;
            count++;
        }

        if (count > 0)
            LOG_DEBUG("LearnedCmd", "Merged " + std::to_string(count) + " learned commands into registry.");
        else
            LOG_DEBUG("LearnedCmd", "No learned commands found to merge.");
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("LearnedCmd", std::string("Failed merging learned commands: ") + e.what());
    }
}

// ====================================================
// Core Dispatch
// ====================================================
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg)
{
    initCommands();

    // --- 1. Normal built-in / plugin command path ---
    auto it = commandMap.find(cmd);
    if (it != commandMap.end())
    {
        LOG_DEBUG("Dispatch", "Running command \"" + cmd + "\" arg=\"" + arg + "\"");
        try {
            return it->second(arg);
        } catch (const std::exception& e) {
            LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
            return {"[Error] Exception while running command: " + cmd,
                    false, sf::Color::Red, "ERR_CMD_EXCEPTION"};
        }
    }

    // --- 2. Attempt learned-command lookup (non-breaking) ---
    try {
        auto learned = g_memoryStorage.findLearnedCommand(cmd);
        if (learned.has_value()) {
            LOG_DEBUG("Dispatch", "Matched learned command: \"" + learned->raw + "\" → \"" + learned->normalized + "\"");
            return dispatchCommand(learned->normalized, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Learned-command lookup failed: ") + e.what());
    }

    // --- 2.5. Record unknown command for analysis ---
    try {
        GRIM::MemoryObject unknown;
        unknown.id = GRIM::MemoryObject::generateUUID();
        unknown.timestamp = std::time(nullptr);
        unknown.source = GRIM::SourceTag::UserText;
        unknown.type = GRIM::TypeTag::UnknownCommand;
        unknown.intent = GRIM::IntentTag::Query;
        unknown.context = GRIM::ContextTag::Conversation;
        unknown.raw = cmd + (arg.empty() ? "" : " " + arg);
        unknown.normalized = normalizeWord(cmd);
        unknown.confidence = 0.4f;
        unknown.tags = {"intercepted", "unclassified", "pending_analysis"};

        g_memoryStorage.storeShortTerm(unknown);
        GRIM::MemoryRouter::dispatch(unknown);

        LOG_DEBUG("Dispatch", "Unknown command recorded for later clarification");
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Failed to record unknown command: ") + e.what());
    }

    // --- 2.7. Clarify unknown ---
    try {
        static std::string lastAsked;
        if (cmd != lastAsked && !cmd.empty()) {
            std::string clarification;
            if (GRIM::PersonalityManager::isStable())
                clarification = "Did you mean \"" + GRIMInput::normalizeCommand(cmd) + "\" or something else?";
            else
                clarification = "I didn’t quite catch that. Can you rephrase?";

            std::string resp = ResponseManager::get(clarification);
            history.push(resp, sf::Color::Yellow);
            Voice::speak(resp, "clarify");

            GRIM::MemoryObject clarify;
            clarify.id = GRIM::MemoryObject::generateUUID();
            clarify.timestamp = std::time(nullptr);
            clarify.source = GRIM::SourceTag::GrimInternal;
            clarify.type = GRIM::TypeTag::Event;
            clarify.intent = GRIM::IntentTag::Query;
            clarify.context = GRIM::ContextTag::Conversation;
            clarify.raw = clarification;
            clarify.normalized = "clarification_prompt";
            clarify.confidence = 0.8f;
            clarify.tags = {"clarify", "dialogue"};

            g_memoryStorage.storeShortTerm(clarify);
            lastAsked = cmd;

            LOG_DEBUG("Dispatch", "Clarifying question issued: " + clarification);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Clarification prompt failed: ") + e.what());
    }

    return {"Unknown command: " + cmd, false, sf::Color::Red, "ERR_UNKNOWN_CMD"};
}

// ====================================================
// handleCommand: central command + NLP hub
// ====================================================
void handleCommand(const std::string& line)
{
    LOG_TRACE("HandleCommand", "START line=\"" + line + "\"");

    auto [cmdRaw, arg] = GRIMInput::parseInput(line);
    LOG_TRACE("HandleCommand", "Parsed → cmdRaw=\"" + cmdRaw + "\" arg=\"" + arg + "\"");

    // Clarification handler
    if (g_pendingClarifyCmd.has_value())
    {
        std::string phrase = g_pendingClarifyCmd.value();
        std::string action = line;

        LOG_DEBUG("LearnedCmd", "User defined \"" + phrase + "\" → \"" + action + "\"");

        try {
            g_memoryStorage.storeLearnedCommand(phrase, action, 0.9f);
            g_pendingClarifyCmd.reset();

            std::string resp = ResponseManager::get("Got it — I'll remember that next time.");
            history.push(resp, sf::Color::Green);
            Voice::speak(resp, "learned");
        } catch (const std::exception& e) {
            LOG_ERROR("LearnedCmd", std::string("Failed to store learned command: ") + e.what());
            std::string resp = ResponseManager::get("I couldn't save that one — try again later.");
            history.push(resp, sf::Color::Red);
            Voice::speak(resp, "error");
        }

        return;
    }

    history.push("> " + line, sf::Color::White);
    CommandResult result;

    // RL pre-dispatch observation
    try {
        nlohmann::json obs = {
            {"input", line},
            {"command_raw", cmdRaw},
            {"argument", arg},
            {"context", longTermMemory}
        };
        nlohmann::json rlRes = GRIM::RL::getAction(obs);
        if (rlRes.contains("suggested_command")) {
            std::string suggested = rlRes["suggested_command"].get<std::string>();
            LOG_DEBUG("RL", "RL suggested override: " + suggested);
            cmdRaw = suggested;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Pre-dispatch RL error: ") + e.what());
    }

    if (commandMap.find(cmdRaw) != commandMap.end())
    {
        LOG_TRACE("HandleCommand", "Direct command match: \"" + cmdRaw + "\"");
        result = dispatchCommand(cmdRaw, arg);
    }
    else
    {
        LOG_TRACE("HandleCommand", "No direct match, running NLP parse...");

        std::string normalizedLine = GRIMInput::normalizeLine(line);
        LOG_TRACE("HandleCommand", "Normalized line=\"" + normalizedLine + "\"");

        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        LOG_TRACE("NLP", "Intent name=\"" + intent.name + "\" matched=" +
            (intent.matched ? "true" : "false") + " slots=" + std::to_string(intent.slots.size()));

        for (const auto& [k, v] : intent.slots)
            LOG_TRACE("NLP", "Slot[" + k + "]=\"" + v + "\"");

        std::string cmd = intent.matched ? intent.name : GRIMInput::normalizeCommand(cmdRaw);

        if (intent.matched)
        {
            std::string slotArg;
            if (intent.slots.count("app") && !intent.slots.at("app").empty())
                slotArg = intent.slots.at("app");
            else if (intent.slots.count("target") && !intent.slots.at("target").empty())
                slotArg = intent.slots.at("target");
            else
                for (const auto& [k, v] : intent.slots)
                    if (!v.empty()) { slotArg = v; break; }

            if (!slotArg.empty())
                arg = GRIMInput::cleanArg(slotArg);
        }

        LOG_TRACE("HandleCommand", "Final dispatch values → cmd=\"" + cmd + "\" arg=\"" + arg + "\"");

        if (cmd == "open_app")
        {
            arg = GRIMInput::cleanArg(arg);
            LOG_DEBUG("OpenApp", "Cleaned arg=\"" + arg + "\"");

            std::string resolved;
            try {
                resolved = aliases::resolve(arg);
            } catch (const std::exception& e) {
                LOG_ERROR("OpenApp", "Alias resolution failed: " + std::string(e.what()));
                resolved.clear();
            }

            if (resolved.empty())
            {
                int bestDist = 3;
                std::string bestAlias;
                for (const auto& [alias, target] : aliases::getAll()) {
                    int dist = GRIMInput::normalizeCommand(arg) == alias ? 0 : 3;
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestAlias = alias;
                        resolved = target;
                    }
                }
                if (!resolved.empty())
                    LOG_DEBUG("OpenApp", "Fuzzy matched \"" + arg + "\" → alias \"" + bestAlias + "\" → " + resolved);
            }

            if (resolved.empty())
            {
                LOG_DEBUG("OpenApp", "No alias found, using raw name: " + arg);
                resolved = arg;
            }

            result = dispatchCommand("open_app", resolved);
        }
        else result = dispatchCommand(cmd, arg);
    }

    if (result.message.empty())
    {
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

    // RL post-dispatch feedback
    try {
        nlohmann::json feedback = {
            {"command", cmdRaw},
            {"argument", arg},
            {"success", result.success},
            {"output", result.message}
        };
        GRIM::RL::getAction(feedback);
    } catch (const std::exception& e) {
        LOG_ERROR("RL", std::string("Post-dispatch RL error: ") + e.what());
    }

    GRIM::ContextManager::recordUsage(cmdRaw);
    GRIM::PersonalityManager::updateAfterCommand(result.success);
    GRIM::DialogueProactive::checkAfterCommand(line, result);

    LOG_TRACE("HandleCommand", "END");
}
