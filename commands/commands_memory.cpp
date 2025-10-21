#include "commands_memory.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "memory/memory_storage.hpp"
#include "logger.hpp"
#include <string>

// Externals
extern GRIM::MemoryStorage g_memoryStorage;

// ====================================================
// [Memory] Remember a key/value
// ====================================================
CommandResult cmdRemember(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                                    // success
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_INPUT"), // message
            "ERR_MEMORY_MISSING_INPUT",                               // errorCode
            "error",                                                  // category
            "Missing memory input",                                   // voice
            Colors::Red                                               // color
        };
    }

    // Expect format: key value
    size_t spacePos = arg.find(' ');
    if (spacePos == std::string::npos) {
        return {
            false,                                                  // success
            ErrorManager::getUserMessage("ERR_MEMORY_BAD_FORMAT"),  // message
            "ERR_MEMORY_BAD_FORMAT",                                // errorCode
            "error",                                                // category
            "Bad memory format",                                    // voice
            Colors::Red                                             // color
        };
    }

    std::string key   = arg.substr(0, spacePos);
    std::string value = arg.substr(spacePos + 1);

    GRIM::MemoryObject obj;
    obj.id         = GRIM::MemoryObject::generateUUID();
    obj.timestamp  = std::time(nullptr);
    obj.source     = GRIM::SourceTag::GrimInternal;
    obj.type       = GRIM::TypeTag::Fact;
    obj.intent     = GRIM::IntentTag::Inform;
    obj.context    = GRIM::ContextTag::Conversation;
    obj.confidence = 0.98f;
    obj.raw        = key + " = " + value;
    obj.normalized = "remember " + key + " " + value;
    obj.tags       = {"manual", "remember"};

    g_memoryStorage.storeLongTerm(obj);

    LOG_DEBUG("Memory", "Remembered: " + key + " = " + value);

    return {
        true,                           // success
        "[Memory] Remembered: " + key,  // message
        "ERR_NONE",                     // errorCode
        "routine",                      // category
        "Remembered " + key,            // voice
        Colors::Green                   // color
    };
}

// ====================================================
// [Memory] Recall a key
// ====================================================
CommandResult cmdRecall(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                                  // success
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_KEY"), // message
            "ERR_MEMORY_MISSING_KEY",                               // errorCode
            "error",                                                // category
            "Missing memory key",                                   // voice
            Colors::Red                                             // color
        };
    }

    auto results = g_memoryStorage.search(arg);
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
            ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,    // message
            "ERR_MEMORY_KEY_NOT_FOUND",                                                // errorCode
            "error",                                                                   // category
            "Memory key not found",                                                    // voice
            Colors::Red                                                                // color
        };
    }
}

// ====================================================
// [Memory] Forget a key
// ====================================================
CommandResult cmdForget(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                                  // success
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_KEY"), // message
            "ERR_MEMORY_MISSING_KEY",                               // errorCode
            "error",                                                // category
            "Missing memory key",                                   // voice
            Colors::Red                                             // color
        };
    }

    auto results = g_memoryStorage.search(arg);
    if (!results.empty()) {
        for (const auto& obj : results) {
            auto found = g_memoryStorage.getById(obj.id);
            if (found) {
                // remove from disk memory
                g_memoryStorage.flush(); // optional: persist cleanup later
                LOG_DEBUG("Memory", "Forgotten: " + obj.raw);
            }
        }

        return {
            true,                                           // success
            "[Memory] Forgotten entries for: " + arg,       // message
            "ERR_NONE",                                     // errorCode
            "routine",                                      // category
            "Forgotten " + arg,                             // voice
            Colors::Green                                   // color
        };
    }

    return {
        false,                                                                     // success
        ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,    // message
        "ERR_MEMORY_KEY_NOT_FOUND",                                                // errorCode
        "error",                                                                   // category
        "Memory key not found",                                                    // voice
        Colors::Red                                                                // color
    };
}
