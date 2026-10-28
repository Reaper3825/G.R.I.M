#include "commands_core.hpp"

#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "ai/ai.hpp"
#include <mutex>
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
    static std::once_flag registered;
    std::call_once(registered, [] {
        LOG_DEBUG("Core", "Ensuring core plugins registered...");
        registerCorePlugin();
    });
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
