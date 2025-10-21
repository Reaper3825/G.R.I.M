#include "commands_interface.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "nlp/nlp.hpp"
#include "resources.hpp"
#include <sstream>
#include <string>

// ====================================================
// [Utility] Clear console
// ====================================================
CommandResult cmdClean([[maybe_unused]] const std::string& arg) {
    return {
        true,                           // success
        "[Utility] Console cleared.",   // message
        "ERR_NONE",                     // errorCode
        "routine",                      // category
        "Console cleared",              // voice
        Colors::Green                   // color
    };
}

// ====================================================
// [Utility] Show help text
// ====================================================
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
        true,               // success
        helpText,           // message
        "ERR_NONE",         // errorCode
        "summary",          // category
        "Help shown",       // voice
        Colors::Cyan        // color
    };
}

// ====================================================
// [Utility] Reload NLP rules
// ====================================================
CommandResult cmd_reloadNLP([[maybe_unused]] const std::string& arg) {
    std::string err;
    if (!g_nlp.load_rules(getResourcePath() + "/nlp_rules.json", &err)) {
        return {
            false,                                      // success
            "[NLP] Failed to reload rules: " + err,     // message
            "ERR_NLP_LOAD_FAILED",                      // errorCode
            "error",                                    // category
            "NLP reload failed",                        // voice
            Colors::Red                                 // color
        };
    }

    return {
        true,                               // success
        "[NLP] Rules reloaded.",            // message
        "ERR_NONE",                         // errorCode
        "routine",                          // category
        "NLP rules reloaded",               // voice
        Colors::Yellow                      // color
    };
}
