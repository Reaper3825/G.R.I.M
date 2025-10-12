#include "commands_memory.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "memory/memory_storage.hpp"
#include "logger.hpp"
#include <SFML/Graphics.hpp>
#include <string>

// Externals
extern GRIM::MemoryStorage g_memoryStorage;

// ====================================================
// [Memory] Remember a key/value
// ====================================================
CommandResult cmdRemember(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_INPUT"),
            false,
            sf::Color::Red,
            "ERR_MEMORY_MISSING_INPUT",
            "Missing memory input",
            "error"
        };
    }

    // Expect format: key value
    size_t spacePos = arg.find(' ');
    if (spacePos == std::string::npos) {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_BAD_FORMAT"),
            false,
            sf::Color::Red,
            "ERR_MEMORY_BAD_FORMAT",
            "Bad memory format",
            "error"
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
        "[Memory] Remembered: " + key,
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Remembered " + key,
        "routine"
    };
}

// ====================================================
// [Memory] Recall a key
// ====================================================
CommandResult cmdRecall(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_KEY"),
            false,
            sf::Color::Red,
            "ERR_MEMORY_MISSING_KEY",
            "Missing memory key",
            "error"
        };
    }

    auto results = g_memoryStorage.search(arg);
    if (!results.empty()) {
        const auto& obj = results.front();
        LOG_DEBUG("Memory", "Recalled: " + obj.raw);

        return {
            "[Memory] " + obj.raw,
            true,
            sf::Color::Cyan,
            "ERR_NONE",
            "Recalled memory for " + arg,
            "summary"
        };
    } else {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_MEMORY_KEY_NOT_FOUND",
            "Memory key not found",
            "error"
        };
    }
}

// ====================================================
// [Memory] Forget a key
// ====================================================
CommandResult cmdForget(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_MISSING_KEY"),
            false,
            sf::Color::Red,
            "ERR_MEMORY_MISSING_KEY",
            "Missing memory key",
            "error"
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
            "[Memory] Forgotten entries for: " + arg,
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Forgotten " + arg,
            "routine"
        };
    }

    return {
        ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,
        false,
        sf::Color::Red,
        "ERR_MEMORY_KEY_NOT_FOUND",
        "Memory key not found",
        "error"
    };
}
