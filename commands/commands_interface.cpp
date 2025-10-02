#include "commands_interface.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "nlp/nlp.hpp"
#include "resources.hpp"

#include <SFML/Graphics.hpp>
#include <sstream>
#include <string>

// ------------------------------------------------------------
// [Utility] Clear console
// ------------------------------------------------------------
CommandResult cmdClean([[maybe_unused]] const std::string& arg) {
    return {
        "[Utility] Console cleared.",
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Console cleared",   // voice
        "routine"            // category
    };
}

// ------------------------------------------------------------
// [Utility] Show help text
// ------------------------------------------------------------
CommandResult cmdShowHelp([[maybe_unused]] const std::string& arg) {
    std::string helpText =
        "[Help] Available commands:\n"
        "- remember <key> <value>\n"
        "- recall <key>\n"
        "- forget <key>\n"
        "- ai_backend <name>\n"
        "- reloadnlp\n"
        "- pwd\n"
        "- cd <dir>\n"
        "- ls\n"
        "- mkdir <dir>\n"
        "- rm <file>\n"
        "- set_timer <seconds>\n"
        "- sysinfo\n"
        "- clean\n"
        "- help\n"
        "- voice\n"
        "- voice_stream\n";

    return {
        helpText,
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Help shown",        // voice
        "summary"            // category
    };
}

// ------------------------------------------------------------
// [Utility] Reload NLP rules
// ------------------------------------------------------------
CommandResult cmd_reloadNLP([[maybe_unused]] const std::string& arg) {
    std::string err;
    if (!g_nlp.load_rules(getResourcePath() + "/nlp_rules.json", &err)) {
        return {
            "[NLP] Failed to reload rules: " + err,
            false,
            sf::Color::Red,
            "ERR_NLP_LOAD_FAILED",
            "NLP reload failed",  // voice
            "error"               // category
        };
    }

    return {
        "[NLP] Rules reloaded.",
        true,
        sf::Color::Yellow,
        "ERR_NONE",
        "NLP rules reloaded",    // voice
        "routine"                // category
    };
}