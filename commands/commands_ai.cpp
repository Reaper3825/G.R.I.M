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
            "[AI] Current backend: " + resolveBackendURL(),
            true,
            sf::Color::Cyan,
            "ERR_NONE",
            "Current AI backend",
            "summary"
        };
    }

    std::string selected = (input == "auto") ? autoSelectBackend() : input;

    if (selected == "ollama" || selected == "localai" || selected == "openai") {
        aiConfig["backend"] = selected;
        LOG_TRACE("AI", "Backend set to: " + selected);

        return {
            "[AI] Backend set to: " + selected,
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Backend set to " + selected,
            "routine"
        };
    }

    LOG_ERROR("AI", "Invalid backend: " + input);
    return {
        ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + input,
        false,
        sf::Color::Red,
        "ERR_AI_INVALID_BACKEND",
        "Invalid backend",
        "error"
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
                return { reply, true, sf::Color::Cyan, "ERR_NONE", reply, "routine" };
            }
        }

        LOG_ERROR("AI", "Ollama backend error");
        return {
            "[AI] Ollama backend error",
            false,
            sf::Color::Red,
            "ERR_AI_BACKEND_FAILED",
            "Ollama backend error",
            "error"
        };
    }

    CommandResult result = ai_process(arg);

    if (result.category.empty()) result.category = "routine";
    if (result.color == sf::Color()) result.color = sf::Color::Cyan;

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
            ErrorManager::getUserMessage("ERR_APP_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_APP_NO_ARGUMENT",
            "Missing application name",
            "error"
        };
    }

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(nullptr, "open", appPath.c_str(), nullptr, nullptr, SW_SHOWNORMAL);

    if ((intptr_t)result <= 32) {
        LOG_ERROR("cmdOpenApp", "ShellExecuteA failed (" + std::to_string((intptr_t)result) + ") for: " + appPath);
        return {
            ErrorManager::getUserMessage("ERR_APP_LAUNCH_FAILED") + ": " + appPath,
            false,
            sf::Color::Red,
            "ERR_APP_LAUNCH_FAILED",
            "Failed to open " + appPath,
            "error"
        };
    }

    LOG_DEBUG("cmdOpenApp", "Successfully launched: " + appPath);
    return { "[App] Launched: " + appPath, true, sf::Color::Green, "ERR_NONE", "Opening " + appPath, "routine" };
#else
    LOG_INFO("cmdOpenApp", "(Stub) Would open: " + appPath);
    return { "[App] (Stub) Would open: " + appPath, true, sf::Color::Green, "ERR_NONE", "Opening " + appPath, "routine" };
#endif
}

CommandResult cmdSearchWeb(const std::string& arg) {
    std::string query = trim(arg);

    if (query.empty()) {
        LOG_DEBUG("Web", "No search query provided");
        return {
            ErrorManager::getUserMessage("ERR_WEB_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_WEB_NO_ARGUMENT",
            "No search query",
            "error"
        };
    }

    std::string url = "https://www.google.com/search?q=" + query;
    LOG_DEBUG("Web", "Searching web for: " + query);

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if ((intptr_t)result <= 32) {
        LOG_ERROR("Web", "Failed to open browser for query: " + query);
        return {
            ErrorManager::getUserMessage("ERR_WEB_OPEN_FAILED") + ": " + query,
            false,
            sf::Color::Red,
            "ERR_WEB_OPEN_FAILED",
            "Web search failed",
            "error"
        };
    }
    return { "[Web] Searching: " + query, true, sf::Color::Cyan, "ERR_NONE", "Searching web for " + query, "routine" };
#else
    return { "[Web] (Stub) Would search for: " + query, true, sf::Color::Cyan, "ERR_NONE", "Searching web for " + query, "routine" };
#endif
}
