#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "system_detect.hpp"
#include "aliases.hpp"   // 🔹 for app alias resolution
#include "nlp.hpp"

#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

// ------------------------------------------------------------
// Externals
// ------------------------------------------------------------
extern nlohmann::json aiConfig;
extern SystemInfo g_systemInfo; // populated at bootstrap

// ------------------------------------------------------------
// Helper: choose best backend automatically
// ------------------------------------------------------------
static std::string autoSelectBackend() {
    if (g_systemInfo.hasGPU && (g_systemInfo.hasCUDA || g_systemInfo.hasROCm || g_systemInfo.hasMetal)) {
        return "localai";
    }
    if (g_systemInfo.osName == "Linux" || g_systemInfo.osName == "macOS") {
        return "ollama";
    }
    return "openai";
}

// ------------------------------------------------------------
// [AI] Select / show current backend
// ------------------------------------------------------------
CommandResult cmdAiBackend(const std::string& arg) {
    if (arg.empty()) {
        return {
            "[AI] Current backend: " + aiConfig.value("backend", "openai"),
            true,
            sf::Color::Cyan,
            "ERR_NONE"
        };
    }

    std::string selected = arg;
    if (arg == "auto") {
        selected = autoSelectBackend();
    }

    if (selected == "ollama" || selected == "localai" || selected == "openai") {
        aiConfig["backend"] = selected;

        std::ofstream f("ai_config.json");
        if (f.is_open()) {
            f << aiConfig.dump(4);
            f.close();
        }

        return {
            "[AI] Backend set to: " + selected,
            true,
            sf::Color::Green,
            "ERR_NONE"
        };
    }

    // invalid backend
    return {
        ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + arg,
        false,
        sf::Color::Red,
        "ERR_AI_INVALID_BACKEND"
    };
}

// ------------------------------------------------------------
// [NLP] Reload rules
// ------------------------------------------------------------
CommandResult cmdReloadNlp(const std::string& arg) {
    return reloadNlpRules();
}

// ------------------------------------------------------------
// [AI] General query (catch-all) → grim_ai
// ------------------------------------------------------------
CommandResult cmdGrimAi(const std::string& arg) {
    std::string query = arg;

    if (query.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_AI_NO_QUERY"),
            false,
            sf::Color::Red,
            "ERR_AI_NO_QUERY"
        };
    }

    std::string backend = aiConfig.value("backend", "openai");
    std::string response;

    try {
        if (backend == "openai") {
            response = "[Stub: OpenAI would answer here]";
        }
        else if (backend == "ollama") {
            response = "[Stub: Ollama would answer here]";
        }
        else if (backend == "localai") {
            response = "[Stub: LocalAI would answer here]";
        }
        else {
            return {
                ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + backend,
                false,
                sf::Color::Red,
                "ERR_AI_INVALID_BACKEND"
            };
        }
    } catch (const std::exception& ex) {
        return {
            ErrorManager::getUserMessage("ERR_AI_QUERY_FAILED") + " (" + ex.what() + ")",
            false,
            sf::Color::Red,
            "ERR_AI_QUERY_FAILED"
        };
    }

    return {
        "[AI] " + response,
        true,
        sf::Color::Cyan,
        "ERR_NONE"
    };
}

// ------------------------------------------------------------
// [Apps] Open local application by alias
// ------------------------------------------------------------
CommandResult cmdOpenApp(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_APP_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_APP_NO_ARGUMENT"
        };
    }

    std::string resolved = resolveAlias(arg);

    if (resolved.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_APP_UNKNOWN_ALIAS") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_APP_UNKNOWN_ALIAS"
        };
    }

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(NULL, "open", resolved.c_str(), NULL, NULL, SW_SHOWNORMAL);
    if ((intptr_t)result <= 32) {
        return {
            ErrorManager::getUserMessage("ERR_APP_LAUNCH_FAILED") + ": " + resolved,
            false,
            sf::Color::Red,
            "ERR_APP_LAUNCH_FAILED"
        };
    }
    return {
        "[App] Opened: " + resolved,
        true,
        sf::Color::Green,
        "ERR_NONE"
    };
#else
    return {
        "[App] (Stub) Would open: " + resolved,
        true,
        sf::Color::Green,
        "ERR_NONE"
    };
#endif
}

// ------------------------------------------------------------
// [Web] Search the web with default browser
// ------------------------------------------------------------
CommandResult cmdSearchWeb(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_WEB_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_WEB_NO_ARGUMENT"
        };
    }

    std::string url = "https://www.google.com/search?q=" + arg;

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
    if ((intptr_t)result <= 32) {
        return {
            ErrorManager::getUserMessage("ERR_WEB_OPEN_FAILED") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_WEB_OPEN_FAILED"
        };
    }
    return {
        "[Web] Searching: " + arg,
        true,
        sf::Color::Cyan,
        "ERR_NONE"
    };
#else
    return {
        "[Web] (Stub) Would search for: " + arg,
        true,
        sf::Color::Cyan,
        "ERR_NONE"
    };
#endif
}
