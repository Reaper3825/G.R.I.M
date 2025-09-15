#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "system_detect.hpp"
#include "nlp.hpp"   // 🔹 use nlp.hpp instead of nlp_rules.hpp

#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>

// Externals
extern nlohmann::json aiConfig;
extern SystemInfo g_systemInfo; // populated at bootstrap

// ------------------------------------------------------------
// Helper: choose best backend automatically
// ------------------------------------------------------------
static std::string autoSelectBackend() {
    // If GPU is available and CUDA/ROCm/Metal are supported → prefer LocalAI
    if (g_systemInfo.hasGPU && (g_systemInfo.hasCUDA || g_systemInfo.hasROCm || g_systemInfo.hasMetal)) {
        return "localai";
    }

    // If system is Linux/macOS with Ollama installed → prefer Ollama
    if (g_systemInfo.osName == "Linux" || g_systemInfo.osName == "macOS") {
        return "ollama";
    }

    // Otherwise → fallback to OpenAI cloud
    return "openai";
}

// ------------------------------------------------------------
// [AI] Select / show current backend
// ------------------------------------------------------------
CommandResult cmdAiBackend(const std::string& arg) {
    if (arg.empty()) {
        // No argument → show current backend
        return {
            "[AI] Current backend: " + aiConfig["backend"].get<std::string>(),
            true,
            sf::Color::Cyan,
            "" // no error code
        };
    }

    std::string selected = arg;

    if (arg == "auto") {
        selected = autoSelectBackend();
    }

    if (selected == "auto" || selected == "ollama" || selected == "localai" || selected == "openai") {
        // Save backend choice
        aiConfig["backend"] = selected;

        // Persist change
        std::ofstream f("ai_config.json");
        if (f.is_open()) {
            f << aiConfig.dump(4);
            f.close();
        }

        return {
            "[AI] Backend set to: " + selected,
            true,
            sf::Color::Green,
            "" // no error code
        };
    }

    // Invalid option → return error
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
CommandResult cmdReloadNlp([[maybe_unused]] const std::string& arg) {
    return reloadNlpRules(); // 🔹 now returns CommandResult directly
}
