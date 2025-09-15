#include "commands_memory.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"

#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
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
            "ERR_MEMORY_MISSING_INPUT"
        };
    }

    // Expect format: key value
    size_t spacePos = arg.find(' ');
    if (spacePos == std::string::npos) {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_BAD_FORMAT"),
            false,
            sf::Color::Red,
            "ERR_MEMORY_BAD_FORMAT"
        };
    }

    std::string key = arg.substr(0, spacePos);
    std::string value = arg.substr(spacePos + 1);

    longTermMemory[key] = value;

    // persist memory
    std::ofstream f("memory.json");
    if (f.is_open()) {
        f << longTermMemory.dump(4);
        f.close();
    }

    return {
        "[Memory] Remembered: " + key,
        true,
        sf::Color::Green
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
            "ERR_MEMORY_MISSING_KEY"
        };
    }

    if (longTermMemory.contains(arg)) {
        return {
            "[Memory] " + arg + " = " + longTermMemory[arg].get<std::string>(),
            true,
            sf::Color::Cyan
        };
    } else {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_MEMORY_KEY_NOT_FOUND"
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
            "ERR_MEMORY_MISSING_KEY"
        };
    }

    if (longTermMemory.contains(arg)) {
        longTermMemory.erase(arg);

        // persist memory
        std::ofstream f("memory.json");
        if (f.is_open()) {
            f << longTermMemory.dump(4);
            f.close();
        }

        return {
            "[Memory] Forgotten: " + arg,
            true,
            sf::Color::Green
        };
    } else {
        return {
            ErrorManager::getUserMessage("ERR_MEMORY_KEY_NOT_FOUND") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_MEMORY_KEY_NOT_FOUND"
        };
    }
}
