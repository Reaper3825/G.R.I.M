#include "commands_memory.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "memory/unified_memory.hpp"
#include "logger.hpp"
#include <string>

// Externals
extern GRIM::UnifiedMemoryStorage g_memoryStorage;

// ====================================================
// [Memory] Remember a fact or key/value pair
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

    std::string key;
    std::string value;
    std::string fullText = arg;
    
    // Check for explicit "key is value" or "key = value" pattern
    size_t isPos = arg.find(" is ");
    size_t equalsPos = arg.find(" = ");
    
    if (isPos != std::string::npos) {
        // Pattern: "my name is austin" -> key="my name", value="austin"
        key = arg.substr(0, isPos);
        value = arg.substr(isPos + 4); // Skip " is "
    } else if (equalsPos != std::string::npos) {
        // Pattern: "name = austin" -> key="name", value="austin"
        key = arg.substr(0, equalsPos);
        value = arg.substr(equalsPos + 3); // Skip " = "
    } else {
        // No clear pattern - store the entire phrase as a fact
        // Use first word as key for searchability
        size_t spacePos = arg.find(' ');
        if (spacePos != std::string::npos) {
            key = arg.substr(0, spacePos);
            value = arg.substr(spacePos + 1);
        } else {
            // Single word - use it as both key and value
            key = arg;
            value = arg;
        }
    }

    GRIM::UnifiedMemoryObject obj;
    obj.id         = GRIM::UnifiedMemoryObject::generateID();
    obj.timestamp  = static_cast<uint64_t>(std::time(nullptr));
    obj.source     = GRIM::SourceType::GRIM_INTERNAL;
    obj.type       = GRIM::TypeTag::FACT;
    obj.intent     = GRIM::MemoryIntent::INFORM;
    obj.context    = GRIM::ContextType::CONVERSATION;
    obj.confidence = 0.98f;
    obj.raw        = fullText;
    obj.normalized = "remember " + fullText;
    obj.tags       = {"manual", "remember"};

    g_memoryStorage.storeLongTerm(obj);

    LOG_DEBUG("Memory", "Remembered: " + fullText);

    return {
        true,                                  // success
        "[Memory] Remembered: " + fullText,    // message
        "ERR_NONE",                            // errorCode
        "routine",                             // category
        "Remembered that " + fullText,         // voice
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
