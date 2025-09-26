#include "commands_ai.hpp" 
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "system_detect.hpp"
#include "aliases.hpp"     // 🔹 for app alias resolution
#include "nlp.hpp"
#include "ai.hpp"

// External libs not in pch.hpp
#include <cpr/cpr.h>       // 🔹 Needed for Ollama HTTP

#ifdef _WIN32
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

        // Config persistence is centralized — just mark in-memory change.
        return {
            "[AI] Backend set to: " + selected,
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Backend set to " + selected,
            "routine"
        };
    }

    return {
        ErrorManager::getUserMessage("ERR_AI_INVALID_BACKEND") + ": " + input,
        false,
        sf::Color::Red,
        "ERR_AI_INVALID_BACKEND",
        "Invalid backend",
        "error"
    };
}

// ------------------------------------------------------------
// [NLP] Reload rules
// ------------------------------------------------------------
CommandResult cmdReloadNlp(const std::string& /*arg*/) {
    CommandResult r = reloadNlpRules();
    if (r.success) {
        r.voice = "NLP rules reloaded";
        r.category = "routine";
    } else {
        r.voice = "Failed to reload NLP rules";
        r.category = "error";
    }
    return r;
}

// ------------------------------------------------------------
// [AI] General query (catch-all) → grim_ai
// ------------------------------------------------------------
CommandResult cmdGrimAi(const std::string& arg) {
    std::cerr << "[AI] cmdGrimAi called with arg=\"" << arg << "\"\n";

    // Resolve backend so logs are clear
    std::string backend = resolveBackendURL();
    std::cerr << "[AI] Current backend resolved: " << backend << "\n";

    // Special handling for Ollama backend
    if (backend == "ollama") {
        std::string model  = aiConfig.value("default_model", "mistral");
        std::string prompt = arg;

        std::string modelCopy = model;
        if (modelCopy.find(':') == std::string::npos) {
            modelCopy += ":latest"; // default tag
        }

        auto resp = cpr::Post(
            cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/generate" },
            cpr::Header{{"Content-Type","application/json"}},
            cpr::Body{ nlohmann::json{
                {"model", modelCopy},
                {"prompt", prompt},
                {"stream", false}   // 🔹 non-streaming mode
            }.dump() }
        );

        if (resp.status_code == 200) {
            auto j = nlohmann::json::parse(resp.text, nullptr, false);
            if (!j.is_discarded() && j.contains("response")) {
                std::string reply = j["response"].get<std::string>();

                return {
                    reply,
                    true,
                    sf::Color::Cyan,
                    "ERR_NONE",
                    reply,   // voice
                    "routine"
                };
            }
        }

        return {
            "[AI] Ollama backend error",
            false,
            sf::Color::Red,
            "ERR_AI_BACKEND_FAILED",
            "Ollama backend error",
            "error"
        };
    }

    // 🔹 Run the default AI pipeline (localai / openai)
    CommandResult result = ai_process(arg);

    // Make sure category + color are consistent
    if (result.category.empty()) result.category = "routine";
    if (result.color == sf::Color()) result.color = sf::Color::Cyan;

    // If the AI failed, report error through ErrorManager
    if (!result.success) {
        std::cerr << "[AI] grim_ai failed with code=" << result.errorCode << "\n";
        return ErrorManager::report(result.errorCode);
    }

    return result;
}

// ------------------------------------------------------------
// [Apps] Open local application by alias
// ------------------------------------------------------------
CommandResult cmdOpenApp(const std::string& arg) {
    std::cerr << "[DEBUG][cmdOpenApp] Received arg=\"" << arg << "\"\n";

    std::string appPath = arg;
    if (appPath.empty()) {
        std::cerr << "[DEBUG][cmdOpenApp] ERROR: empty arg\n";
        return {
            ErrorManager::getUserMessage("ERR_APP_NO_ARGUMENT"),
            false,
            sf::Color::Red,
            "ERR_APP_NO_ARGUMENT"
        };
    }

#ifdef _WIN32
    HINSTANCE result = ShellExecuteA(
        nullptr,
        "open",
        appPath.c_str(),
        nullptr,
        nullptr,
        SW_SHOWNORMAL
    );

    if ((intptr_t)result <= 32) {
        std::cerr << "[DEBUG][cmdOpenApp] ShellExecuteA failed (" 
                  << (intptr_t)result << ") for: " << appPath << "\n";
        return {
            ErrorManager::getUserMessage("ERR_APP_LAUNCH_FAILED") + ": " + appPath,
            false,
            sf::Color::Red,
            "ERR_APP_LAUNCH_FAILED"
        };
    }

    std::cerr << "[DEBUG][cmdOpenApp] Successfully launched: " << appPath << "\n";
    return {
        "[App] Launched: " + appPath,
        true,
        sf::Color::Green,
        "ERR_NONE"
    };
#else
    // Linux / macOS stub
    std::cerr << "[DEBUG][cmdOpenApp] (Stub) Would open: " << appPath << "\n";
    return {
        "[App] (Stub) Would open: " + appPath,
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
            "ERR_WEB_NO_ARGUMENT",
            "No search query",
            "error"
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
            "ERR_WEB_OPEN_FAILED",
            "Web search failed",
            "error"
        };
    }
    return {
        "[Web] Searching: " + query,
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Searching web for " + query,
        "routine"
    };
#else
    // Linux/macOS stub
    return {
        "[Web] (Stub) Would search for: " + query,
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Searching web for " + query,
        "routine"
    };
#endif
}
