#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "../MMO/Core/HardwareInventory.hpp"
#include "aliases.hpp"
#include "nlp/nlp.hpp"
#include "ai/ai.hpp"
#include "logger.hpp"

#ifdef _WIN32
  #include <shellapi.h>
#endif

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

CommandResult cmdOpenApp(const std::string& arg) {
    LOG_DEBUG("cmdOpenApp", "Received arg=\"" + arg + "\"");

    std::string appPath = arg;
    if (appPath.empty()) {
        LOG_DEBUG("cmdOpenApp", "Empty argument");
        return {
            false,                                                  // success
            ErrorManager::getUserMessage("ERR_APP_NO_ARGUMENT"),    // message
            "ERR_APP_NO_ARGUMENT",                                  // errorCode
            "error",                                                // category
            "Missing application name",                             // voice
            Colors::Red                                             // color
        };
    }

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(nullptr, "open", appPath.c_str(), nullptr, nullptr, SW_SHOWNORMAL);

    if ((intptr_t)result <= 32) {
        LOG_ERROR("cmdOpenApp", "ShellExecuteA failed (" + std::to_string((intptr_t)result) + ") for: " + appPath);
        return {
            false,                                                                         // success
            ErrorManager::getUserMessage("ERR_APP_LAUNCH_FAILED") + ": " + appPath,       // message
            "ERR_APP_LAUNCH_FAILED",                                                       // errorCode
            "error",                                                                       // category
            "Failed to open " + appPath,                                                   // voice
            Colors::Red                                                                    // color
        };
    }

    LOG_DEBUG("cmdOpenApp", "Successfully launched: " + appPath);
    return {
        true,                               // success
        "[App] Launched: " + appPath,       // message
        "ERR_NONE",                         // errorCode
        "routine",                          // category
        "Opening " + appPath,               // voice
        Colors::Green                       // color
    };
#else
    LOG_DEBUG("cmdOpenApp", "(Stub) Would open: " + appPath);
    return {
        true,                                           // success
        "[App] (Stub) Would open: " + appPath,          // message
        "ERR_NONE",                                     // errorCode
        "routine",                                      // category
        "Opening " + appPath,                           // voice
        Colors::Green                                   // color
    };
#endif
}

CommandResult cmdSearchWeb(const std::string& arg) {
    std::string query = trim(arg);

    if (query.empty()) {
        LOG_DEBUG("Web", "No search query provided");
        return {
            false,                                                  // success
            ErrorManager::getUserMessage("ERR_WEB_NO_ARGUMENT"),    // message
            "ERR_WEB_NO_ARGUMENT",                                  // errorCode
            "error",                                                // category
            "No search query",                                      // voice
            Colors::Red                                             // color
        };
    }

    std::string url = "https://www.google.com/search?q=" + query;
    LOG_DEBUG("Web", "Searching web for: " + query);

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if ((intptr_t)result <= 32) {
        LOG_ERROR("Web", "Failed to open browser for query: " + query);
        return {
            false,                                                                     // success
            ErrorManager::getUserMessage("ERR_WEB_OPEN_FAILED") + ": " + query,       // message
            "ERR_WEB_OPEN_FAILED",                                                     // errorCode
            "error",                                                                   // category
            "Web search failed",                                                       // voice
            Colors::Red                                                                // color
        };
    }
    return {
        true,                                   // success
        "[Web] Searching: " + query,            // message
        "ERR_NONE",                             // errorCode
        "routine",                              // category
        "Searching web for " + query,           // voice
        Colors::Cyan                            // color
    };
#else
    return {
        true,                                           // success
        "[Web] (Stub) Would search for: " + query,      // message
        "ERR_NONE",                                     // errorCode
        "routine",                                      // category
        "Searching web for " + query,                   // voice
        Colors::Cyan                                    // color
    };
#endif
}
