#include "memory_provider.hpp"
#include "plugins/plugin_logging.hpp"

// ====================================================
// [Memory] Remember a fact or key/value pair
// ====================================================
CommandResult cmdRemember(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                                    // success
            memoryProvider::errorMessage("ERR_MEMORY_MISSING_INPUT"), // message
            "ERR_MEMORY_MISSING_INPUT",                               // errorCode
            "error",                                                  // category
            "Missing memory input",                                   // voice
            Colors::Red                                               // color
        };
    }

    memoryProvider::stageManual(arg);

    LOG_DEBUG("Memory", "Remembered: " + arg);

    return {
        true,                                  // success
        "[Memory] Remembered: " + arg,    // message
        "ERR_NONE",                            // errorCode
        "routine",                             // category
        "Remembered that " + arg,         // voice
        Colors::Green                          // color
    };
}

// ====================================================
// [Memory] Recall a key
// ====================================================
CommandResult cmdRecall(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                                  // success
            memoryProvider::errorMessage("ERR_MEMORY_MISSING_KEY"), // message
            "ERR_MEMORY_MISSING_KEY",                               // errorCode
            "error",                                                // category
            "Missing memory key",                                   // voice
            Colors::Red                                             // color
        };
    }

    auto results = memoryProvider::search(arg);
    if (!results.empty()) {
        const auto& obj = results.front();
        LOG_DEBUG("Memory", "Recalled: " + obj.raw);

        return {
            true,                               // success
            "[Memory] " + obj.raw,              // message
            "ERR_NONE",                         // errorCode
            "summary",                          // category
            "Recalled memory for " + arg,       // voice
            Colors::Cyan                        // color
        };
    } else {
        return {
            false,                                                                     // success
            memoryProvider::errorMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,    // message
            "ERR_MEMORY_KEY_NOT_FOUND",                                                // errorCode
            "error",                                                                   // category
            "Memory key not found",                                                    // voice
            Colors::Red                                                                // color
        };
    }
}
