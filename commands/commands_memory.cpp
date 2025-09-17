#include "commands_memory.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"

#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <string>

// Externals
extern nlohmann::json longTermMemory;

// ------------------------------------------------------------
// [Memory] Remember a key/value
// ------------------------------------------------------------
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

    std::string key = arg.substr(0, spacePos);
    std::string value = arg.substr(spacePos + 1);

    longTermMemory[key] = value;

    return {
        "[Memory] Remembered: " + key,
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Remembered " + key,
        "routine"
    };
}

// ------------------------------------------------------------
// [Memory] Recall a key
// ------------------------------------------------------------
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

    if (longTermMemory.contains(arg)) {
        std::string value = longTermMemory[arg].get<std::string>();
        return {
            "[Memory] " + arg + " = " + value,
            true,
            sf::Color::Cyan,
            "ERR_NONE",
            arg + " is " + value,
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

// ------------------------------------------------------------
// [Memory] Forget a key
// ------------------------------------------------------------
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

    if (longTermMemory.contains(arg)) {
        longTermMemory.erase(arg);

        return {
            "[Memory] Forgotten: " + arg,
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Forgotten " + arg,
            "routine"
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
