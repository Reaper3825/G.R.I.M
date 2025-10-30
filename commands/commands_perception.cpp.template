#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "system_detect.hpp"
#include "aliases.hpp"
#include "nlp/nlp.hpp"
#include "ai/ai.hpp"
#include "logger.hpp"  // ✅ Added

#include <cpr/cpr.h>

#ifdef _WIN32
  #include <shellapi.h>
#endif

extern nlohmann::json aiConfig;
extern SystemInfo g_systemInfo;

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    auto end   = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static std::string autoSelectBackend() {
    if (g_systemInfo.hasGPU && (g_systemInfo.hasCUDA || g_systemInfo.hasROCm || g_systemInfo.hasMetal)) {
        return "localai";
    }
    if (g_systemInfo.osName == "Linux" || g_systemInfo.osName == "macOS") {
        return "ollama";
    }
    return "openai";
}

CommandResult cmdAiBackend(const std::string& arg) {
    std::string input = trim(arg);

    if (input.empty()) {
        return {
            true,                                           // success
            "[AI] Current backend: " + resolveBackendURL(), // message
            "ERR_NONE",                                     // errorCode
            "summary",                                      // category
            "Current AI backend",                           // voice
            Colors::Cyan                                    // color
        };
    }

    std::string selected = (input == "auto") ? autoSelectBackend() : input;

    if (selected == "ollama" || selected == "localai" || selected == "openai") {
        aiConfig["backend"] = selected;
        LOG_TRACE("AI", "Backend set to: " + selected);

        return {
            true,                                       // success
            "[AI] Backend set to: " + selected,         // message
            "ERR_NONE",                                 // errorCode
            "routine",                                  // category
            "Backend set to " + selected,               // voice
            Colors::Green                               // color
        };
    }

    LOG_ERROR("AI", "Invalid backend: " + input);
    return {
        false,                                                                     // success
        ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + input,    // message
        "ERR_AI_INVALID_BACKEND",                                                  // errorCode
        "error",                                                                   // category
        "Invalid backend",                                                         // voice
        Colors::Red                                                                // color
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

CommandResult cmdGrimAi(const std::string& arg) {
    LOG_TRACE("AI", "cmdGrimAi called with arg=\"" + arg + "\"");

    std::string backend = resolveBackendURL();
    LOG_DEBUG("AI", "Current backend resolved: " + backend);

    if (backend == "ollama") {
        std::string model  = aiConfig.value("default_model", "mistral");
        std::string prompt = arg;
        std::string modelCopy = model;
        if (modelCopy.find(':') == std::string::npos) modelCopy += ":latest";

        auto resp = cpr::Post(
            cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/generate" },
            cpr::Header{{"Content-Type","application/json"}},
            cpr::Body{ nlohmann::json{
                {"model", modelCopy},
                {"prompt", prompt},
                {"stream", false}
            }.dump() }
        );

        if (resp.status_code == 200) {
            auto j = nlohmann::json::parse(resp.text, nullptr, false);
            if (!j.is_discarded() && j.contains("response")) {
                std::string reply = j["response"].get<std::string>();
                LOG_TRACE("AI", "Ollama replied successfully");
                return {
                    true,               // success
                    reply,              // message
                    "ERR_NONE",         // errorCode
                    "routine",          // category
                    reply,              // voice
                    Colors::Cyan        // color
                };
            }
        }

        LOG_ERROR("AI", "Ollama backend error");
        return {
            false,                              // success
            "[AI] Ollama backend error",        // message
            "ERR_AI_BACKEND_FAILED",            // errorCode
            "error",                            // category
            "Ollama backend error",             // voice
            Colors::Red                         // color
        };
    }

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
    LOG_INFO("cmdOpenApp", "(Stub) Would open: " + appPath);
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
