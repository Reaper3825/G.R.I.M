#include "commands_ai.hpp"
#include "error_manager.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "aliases.hpp"
#include "nlp/nlp.hpp"
#include "ai/ai.hpp"
#include "logger.hpp"

extern nlohmann::json aiConfig;
extern GRIM::MMO::HardwareInventory g_hardwareInventory;

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    auto end   = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

CommandResult cmdAiBackend(const std::string& arg) {
    std::string input = trim(arg);

    if (input.empty()) {
        std::string current = aiConfig.value("backend", "(not set)");
        return {
            true,
            "[AI] Current backend config: " + current,
            "ERR_NONE",
            "summary",
            "Current AI backend",
            Colors::Cyan
        };
    }

    // All routing is handled by MMO ModelRouter — this just sets the config key
    aiConfig["backend"] = input;
    LOG_TRACE("AI", "Backend config set to: " + input);

    return {
        true,
        "[AI] Backend config set to: " + input,
        "ERR_NONE",
        "routine",
        "Backend set to " + input,
        Colors::Green
    };
}

CommandResult cmdReloadNlp(const std::string& /*arg*/) {
    LOG_PHASE("AI", "Reloading NLP rules");
    CommandResult r = reloadNlpRules();
    if (r.success) {
        r.voice = "NLP rules reloaded";
        r.category = "routine";
    } else {
        LOG_ERROR("AI", "Failed to reload NLP rules");
        r.voice = "Failed to reload NLP rules";
        r.category = "error";
    }
    return r;
}

CommandResult cmdClearContext(const std::string& /*arg*/) {
    clearConversationHistory();
    return {
        true,
        "[AI] Conversation context cleared",
        "ERR_NONE",
        "routine",
        "Context cleared",
        Colors::Green
    };
}

CommandResult cmdGrimAi(const std::string& arg) {
    LOG_TRACE("AI", "cmdGrimAi called with arg=\"" + arg + "\"");

    // All routing via MMO Orchestrator through ai_process
    CommandResult result = ai_process(arg);

    if (result.category.empty()) result.category = "routine";
    if (result.color.r == 255 && result.color.g == 255 && 
        result.color.b == 255 && result.color.a == 255) {
        result.color = Colors::Cyan;
    }

    if (!result.success) {
        LOG_ERROR("AI", "grim_ai failed with code=" + result.errorCode);
        return ErrorManager::report(result.errorCode);
    }

    return result;
}
