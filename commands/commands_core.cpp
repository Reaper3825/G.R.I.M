#include "commands_core.hpp"

#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "ai/ai.hpp"
#include "core/plugin.hpp"
#include "error_manager.hpp"
#include "helpers/color.hpp"
#include "logger.hpp"
#include "voice/voice_speak.hpp"

namespace {
constexpr const char* kDefaultSession = "default";
}

// Forward declaration (defined in core_plugin.cpp).
extern void registerCorePlugin();

bool hasPendingFeedback() {
    return GRIM::MMO::SessionContextManager::instance()
        .getPending(kDefaultSession)
        .has_value();
}

void ensureCorePluginsRegistered() {
    static bool done = false;
    if (done) {
        return;
    }

    done = true;
    LOG_DEBUG("Core", "Ensuring core plugins registered...");
    registerCorePlugin();
}

// Internal execution boundary. Only an already-selected registered tool ID is
// accepted here; unknown IDs fail closed without reinterpretation or learning.
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg) {
    ensureCorePluginsRegistered();

    const auto it = commandMap.find(cmd);
    if (it == commandMap.end()) {
        LOG_DEBUG("Dispatch", "Rejected unknown tool ID: \"" + cmd + "\"");
        return CommandResult{
            false,
            "Unknown command: " + cmd,
            "ERR_UNKNOWN_CMD",
            "",
            "",
            Colors::Red
        };
    }

    LOG_DEBUG("Dispatch", "Running command \"" + cmd + "\" arg=\"" + arg + "\"");
    try {
        CommandResult result = it->second(arg);
        if (result.success) {
            GRIM::MMO::ToolRegistry::instance().recordSuccess(cmd);
        } else {
            GRIM::MMO::ToolRegistry::instance().recordFailure(cmd);
        }
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Dispatch", "Exception in command \"" + cmd + "\": " + e.what());
        GRIM::MMO::ToolRegistry::instance().recordFailure(cmd);
        return CommandResult{
            false,
            "[Error] Exception while running command: " + cmd,
            "ERR_CMD_EXCEPTION",
            "",
            "",
            Colors::Red
        };
    }
}

// User-input boundary. Raw user text is never interpreted as an application
// command; the reasoning model owns that decision.
CommandResult handleCommand(const std::string& line) {
    return handleCommand(line, kDefaultSession);
}

CommandResult handleCommand(
    const std::string& line,
    const std::string& session_id,
    std::optional<GRIM::ReasoningState> reasoning_state) {
    LOG_TRACE("HandleCommand", "START raw model input");

    CommandResult result = ai_process(line, session_id, std::move(reasoning_state));

    if (result.message.empty()) {
        result.message = "[AI] Failed to process request";
        result.success = false;
        if (result.errorCode.empty()) {
            result.errorCode = "ERR_AI_BACKEND_UNAVAILABLE";
        }
    }

    Logger::logResult(result);

    if (!result.voice.empty() && result.voice.find("[TRACE]") == std::string::npos) {
        Voice::speak(
            result.voice,
            result.category.empty() ? "routine" : result.category);
    }

    LOG_TRACE("HandleCommand", "END raw model input");
    return result;
}
