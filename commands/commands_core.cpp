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
#include "input_parser.hpp"
#include "ai/ai.hpp"
#include "core/plugin.hpp" // 🔹 Plugin system
#include <crtdbg.h>

#define CHECK_HEAP() _CrtCheckMemory()

using Voice::speak;

// ====================================================
// Globals
// ====================================================

static std::optional<std::string> g_pendingClarifyCmd;
static std::optional<std::string> g_pendingFeedbackCmd;

// Externals
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;
extern NLP g_nlp;
#define history getConsoleHistory()
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
// Core Plugin Loader — ensures built-in commands exist
// ====================================================
void ensureCorePluginsRegistered()
{
    static bool done = false;
    if (done) return;
    done = true;

    LOG_DEBUG("Core", "Ensuring core plugins registered...");

    // Core built-in plugin (terminal-like commands)
    extern void registerCorePlugin();
    registerCorePlugin();
}

// ====================================================
// Core Dispatch
// ====================================================
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg)
{
    ensureCorePluginsRegistered();

    // ====================================================
    // 1. Built-in / Plugin command path
    // ====================================================
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

    // ====================================================
    // 2. Learned-command lookup
    // ====================================================
    try {
        auto learned = g_memoryStorage.findLearnedCommand(cmd);
        if (learned.has_value()) {
            LOG_DEBUG("Dispatch", "Matched learned command: \"" + learned->raw + "\" → \"" + learned->normalized + "\"");
            return dispatchCommand(learned->normalized, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("Learned-command lookup failed: ") + e.what());
    }

    // ====================================================
    // 3. Record unknown command
    // ====================================================
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

    // ====================================================
    // 4. Try RL bridge reasoning
    // ====================================================
    try {
        nlohmann::json obs = {
            {"type","unknown_command"},
            {"input", cmd + " " + arg},
            {"context", longTermMemory}
        };

        auto rlRes = GRIM::RL::getAction(obs);
        if (rlRes.contains("suggested_command")) {
            std::string inferred = rlRes["suggested_command"].get<std::string>();
            LOG_DEBUG("RL", "RL suggested command: " + inferred);
            return dispatchCommand(inferred, arg);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("RL reasoning failed: ") + e.what());
    }

    // ====================================================
    // 5. AI reasoning fallback — try to infer or converse
    // ====================================================
    try {
        LOG_TRACE("Dispatch", "Calling ai_interpret() for unknown command...");
        CommandResult aiRes = ai_interpret(cmd + " " + arg, true);

        if (aiRes.success && aiRes.category == "command_infer")
        {
            std::string inferred = GRIMInput::normalizeCommand(aiRes.message);
            LOG_DEBUG("AI", "AI inferred new command: " + inferred);

            g_memoryStorage.storeLearnedCommand(cmd, inferred, 0.75f);
            g_learnedCommandMap[cmd] = inferred;
            commandMap[cmd] = handleLearnedCommand;
            CHECK_HEAP();
            std::string resp = ResponseManager::get(
                "Got it — I've learned that \"" + cmd + "\" means \"" + inferred + "\".");
            history.push(resp, sf::Color::Green);
            Voice::speak(resp, "learned");

            return {"Learned new command: " + inferred, true, sf::Color::Green, "ERR_NONE"};
        }
        else if (aiRes.success && aiRes.category == "conversation")
        {
            LOG_DEBUG("AI", "AI interpret returned conversational intent.");
            return aiRes;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", std::string("AI fallback failed: ") + e.what());
    }

    // ====================================================
    // 6. Clarification fallback
    // ====================================================
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

    // === Clarification handler ===
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

    // === Feedback handler ===
    if (g_pendingFeedbackCmd.has_value())
    {
        std::string cmd = g_pendingFeedbackCmd.value();
        std::string answer = GRIMInput::normalizeCommand(line);
        answer.erase(std::remove_if(answer.begin(), answer.end(), ::ispunct), answer.end());
        bool positive = (answer == "yes" || answer == "y" || answer == "yeah" || answer == "correct");
        bool negative = (answer == "no" || answer == "n" || answer == "nope" || answer == "wrong");

        if (positive || negative)
        {
            try {
                nlohmann::json fb = {
                    {"command", cmd},
                    {"feedback", positive ? "positive" : "negative"},
                    {"user_text", line}
                };
                GRIM::RL::getAction(fb);
                LOG_DEBUG("Feedback", "User feedback recorded for \"" + cmd + "\": " + (positive ? "positive" : "negative"));
            } catch (const std::exception& e) {
                LOG_ERROR("Feedback", std::string("Error sending feedback to RL: ") + e.what());
            }

            std::string resp = ResponseManager::get(positive ? "Got it — I'll keep doing that."
                                                             : "Understood — I’ll adjust next time.");
            history.push(resp, positive ? sf::Color::Green : sf::Color::Yellow);
            Voice::speak(resp, "feedback");
            g_pendingFeedbackCmd.reset();
            return;
        }
    }

    // ====================================================
    // Normal execution flow
    // ====================================================
    history.push("> " + line, sf::Color::White);
    CommandResult result;

    // --- RL Pre-dispatch observation ---
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

    // --- Dispatch ---
    if (commandMap.find(cmdRaw) != commandMap.end())
    {
        result = dispatchCommand(cmdRaw, arg);
    }
    else
    {
        std::string normalizedLine = GRIMInput::normalizeLine(line);
        Intent intent = g_nlp.parse(normalizedLine);
        g_lastIntent = intent;

        std::string cmd = intent.matched ? intent.name : GRIMInput::normalizeCommand(cmdRaw);

        if (intent.matched && intent.slots.size())
        {
            for (const auto& [k, v] : intent.slots)
                if (!v.empty()) { arg = GRIMInput::cleanArg(v); break; }
        }

        if (commandMap.find(cmd) != commandMap.end())
            result = dispatchCommand(cmd, arg);
        else
            result = ai_interpret(line, true);
    }

    if (result.message.empty()) {
        result.message = "[no response configured]";
        result.success = false;
        if (result.errorCode.empty()) result.errorCode = "ERR_NONE";
    }

    std::string finalText = ResponseManager::get(result.message);
    Logger::logResult(result);
    history.push(finalText, result.color);
    std::cout << finalText << std::endl;

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos)
        Voice::speak(result.voice, result.category.empty() ? "routine" : result.category);

    if (!g_pendingFeedbackCmd.has_value() && result.success && result.category != "conversation")
    {
        std::string ask = ResponseManager::get("Was that what you wanted?");
        history.push(ask, sf::Color::Yellow);
        Voice::speak(ask, "feedback");
        g_pendingFeedbackCmd = cmdRaw;
        LOG_DEBUG("Feedback", "Opened feedback prompt for command: " + cmdRaw);
    }

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
