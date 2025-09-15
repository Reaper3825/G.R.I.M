#include "commands_ai.hpp"
#include "response_manager.hpp"
#include "voice_speak.hpp"
#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>

// Externals
extern nlohmann::json aiConfig;

CommandResult cmdAiBackend(const std::string& arg) {
    if (arg.empty()) {
        return { "[AI] Current backend: " + aiConfig["backend"].get<std::string>(), true, sf::Color::Cyan };
    }

    if (arg == "auto" || arg == "ollama" || arg == "localai" || arg == "openai") {
        aiConfig["backend"] = arg;

        // persist
        std::ofstream f("ai_config.json");
        f << aiConfig.dump(4);
        f.close();

        return { "[AI] Backend set to: " + arg, true, sf::Color::Green };
    }

    return { "[AI] Invalid backend: " + arg, false, sf::Color::Red };
}

CommandResult cmdReloadNlp(const std::string& arg) {
    // Could call your existing NLP reload function here
    return { "[NLP] Rules reloaded.", true, sf::Color::Green };
}
