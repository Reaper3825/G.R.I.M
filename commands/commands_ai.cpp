#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "system_detect.hpp"
#include "aliases.hpp"     // 🔹 for app alias resolution
#include "nlp.hpp"

#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

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
// Helpers
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// [AI] Select / show current backend
// ------------------------------------------------------------
CommandResult cmdAiBackend(const std::string& arg) {
    std::string input = trim(arg);

    if (input.empty()) {
        return {
            "[AI] Current backend: " + aiConfig.value("backend", "openai"),
            true,
            sf::Color::Cyan,
            "ERR_NONE"
        };
    }

    std::string selected = (input == "auto") ? autoSelectBackend() : input;

    if (selected == "ollama" || selected == "localai" || selected == "openai") {
        aiConfig["backend"] = selected;

        std::ofstream f("ai_config.json");
        if (f.is_open()) {
            f << aiConfig.dump(4);
        }

        return {
            "[AI] Backend set to: " + selected,
            true,
            sf::Color::Green,
            "ERR_NONE"
        };
    }

    return {
        ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + input,
        false,
        sf::Color::Red,
        "ERR_AI_INVALID_BACKEND"
    };
}

// ------------------------------------------------------------
// [NLP] Reload rules
// ------------------------------------------------------------
CommandResult cmdReloadNlp(const std::string& /*arg*/) {
    return reloadNlpRules();
}

// ------------------------------------------------------------
// [AI] General query (catch-all) → grim_ai
// ------------------------------------------------------------
CommandResult cmdGrimAi(const std::string& arg) {
    std::string query = trim(arg);

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
        } else if (backend == "ollama") {
            response = "[Stub: Ollama would answer here]";
        } else if (backend == "localai") {
            response = "[Stub: LocalAI would answer here]";
        } else {
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
// ------------------------------------------------------------
// [Apps] Open local application by alias
// ------------------------------------------------------------
// ------------------------------------------------------------
// [Apps] Open local application by alias
// ------------------------------------------------------------
CommandResult cmdOpenApp(const std::string& arg) {
    std::cout << "[DEBUG][cmdOpenApp] Received arg=\"" << arg << "\"\n";

    std::string appName = arg;
    // Trim leading/trailing whitespace
    auto start = appName.find_first_not_of(" \t\n\r");
    auto end   = appName.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) {
        appName.clear();
    } else {
        appName = appName.substr(start, end - start + 1);
    }

    if (appName.empty()) {
        std::cout << "[DEBUG][cmdOpenApp] appName is EMPTY after trim\n";
        return {
            ErrorManager::getUserMessage("ERR_APP_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_APP_NO_ARGUMENT"
        };
    }

    std::cout << "[DEBUG][cmdOpenApp] After trim, appName=\"" << appName << "\"\n";

    std::string resolved = resolveAlias(appName);
    if (resolved.empty()) {
        std::cout << "[DEBUG][cmdOpenApp] Alias not found for \"" << appName << "\"\n";
        return {
            ErrorManager::getUserMessage("ERR_APP_UNKNOWN_ALIAS") + ": " + appName,
            false,
            sf::Color::Red,
            "ERR_APP_UNKNOWN_ALIAS"
        };
    }

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(
        nullptr, "open", resolved.c_str(),
        nullptr, nullptr, SW_SHOWNORMAL
    );
    if ((intptr_t)result <= 32) {
        return {
            ErrorManager::getUserMessage("ERR_APP_LAUNCH_FAILED") + ": " + resolved,
            false,
            sf::Color::Red,
            "ERR_APP_LAUNCH_FAILED"
        };
    }
    return {
        "[App] Launched: " + resolved,
        true,
        sf::Color::Green,
        "ERR_NONE"
    };
#else
    // Linux/macOS stub
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
    std::string query = trim(arg);

    if (query.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_WEB_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_WEB_NO_ARGUMENT"
        };
    }

    std::string url = "https://www.google.com/search?q=" + query;

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(
        nullptr, "open", url.c_str(),
        nullptr, nullptr, SW_SHOWNORMAL
    );
    if ((intptr_t)result <= 32) {
        return {
            ErrorManager::getUserMessage("ERR_WEB_OPEN_FAILED") + ": " + query,
            false,
            sf::Color::Red,
            "ERR_WEB_OPEN_FAILED"
        };
    }
    return {
        "[Web] Searching: " + query,
        true,
        sf::Color::Cyan,
        "ERR_NONE"
    };
#else
    // Linux/macOS stub
    return {
        "[Web] (Stub) Would search for: " + query,
        true,
        sf::Color::Cyan,
        "ERR_NONE"
    };
#endif
}
