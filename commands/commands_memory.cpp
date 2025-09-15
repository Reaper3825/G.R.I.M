#include "commands_memory.hpp"
#include "console_history.hpp"
#include "response_manager.hpp"
#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <fstream>
#include <iostream>

// Externals
extern nlohmann::json longTermMemory;
extern ConsoleHistory history;

CommandResult cmdRemember(const std::string& arg) {
    if (arg.empty()) return { "[Memory] Nothing to remember.", false, sf::Color::Red };

    // Expect format: key value
    size_t spacePos = arg.find(' ');
    if (spacePos == std::string::npos) return { "[Memory] Usage: remember <key> <value>", false, sf::Color::Red };

    std::string key = arg.substr(0, spacePos);
    std::string value = arg.substr(spacePos + 1);

    longTermMemory[key] = value;

    // persist memory
    std::ofstream f("memory.json");
    f << longTermMemory.dump(4);
    f.close();

    return { "[Memory] Remembered: " + key, true, sf::Color::Green };
}

CommandResult cmdRecall(const std::string& arg) {
    if (arg.empty()) return { "[Memory] Usage: recall <key>", false, sf::Color::Red };

    if (longTermMemory.contains(arg)) {
        return { "[Memory] " + arg + " = " + longTermMemory[arg].get<std::string>(), true, sf::Color::Cyan };
    } else {
        return { "[Memory] Key not found: " + arg, false, sf::Color::Red };
    }
}

CommandResult cmdForget(const std::string& arg) {
    if (arg.empty()) return { "[Memory] Usage: forget <key>", false, sf::Color::Red };

    if (longTermMemory.contains(arg)) {
        longTermMemory.erase(arg);

        // persist memory
        std::ofstream f("memory.json");
        f << longTermMemory.dump(4);
        f.close();

        return { "[Memory] Forgotten: " + arg, true, sf::Color::Green };
    } else {
        return { "[Memory] Key not found: " + arg, false, sf::Color::Red };
    }
}
