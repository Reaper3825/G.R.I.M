#include "commands_filesystem.hpp"
#include "console_history.hpp"
#include <filesystem>
#include <iostream>
#include <SFML/Graphics.hpp>

extern ConsoleHistory history;
extern std::filesystem::path g_currentDir;

CommandResult cmdShowPwd([[maybe_unused]] const std::string& arg) {
    return { "[FS] Current directory: " + g_currentDir.string(), true, sf::Color::Cyan };
}

CommandResult cmdChangeDir(const std::string& arg) {
    if (arg.empty()) return { "[FS] Usage: cd <directory>", false, sf::Color::Red };

    std::filesystem::path newPath = g_currentDir / arg;
    if (!std::filesystem::exists(newPath)) return { "[FS] Directory does not exist: " + arg, false, sf::Color::Red };

    g_currentDir = std::filesystem::canonical(newPath);
    return { "[FS] Changed directory to: " + g_currentDir.string(), true, sf::Color::Green };
}

CommandResult cmdListDir([[maybe_unused]] const std::string& arg) {
    std::string output = "[FS] Contents:\n";
    for (const auto& entry : std::filesystem::directory_iterator(g_currentDir)) {
        output += " - " + entry.path().filename().string() + "\n";
    }
    return { output, true, sf::Color::Cyan };
}

CommandResult cmdMakeDir(const std::string& arg) {
    if (arg.empty()) return { "[FS] Usage: mkdir <directory>", false, sf::Color::Red };

    std::filesystem::path newDir = g_currentDir / arg;
    if (std::filesystem::create_directory(newDir)) {
        return { "[FS] Directory created: " + newDir.string(), true, sf::Color::Green };
    } else {
        return { "[FS] Failed to create directory: " + newDir.string(), false, sf::Color::Red };
    }
}

CommandResult cmdRemoveFile(const std::string& arg) {
    if (arg.empty()) return { "[FS] Usage: rm <file>", false, sf::Color::Red };

    std::filesystem::path file = g_currentDir / arg;
    if (!std::filesystem::exists(file)) return { "[FS] File not found: " + arg, false, sf::Color::Red };

    if (std::filesystem::remove(file)) {
        return { "[FS] Removed: " + file.string(), true, sf::Color::Green };
    } else {
        return { "[FS] Failed to remove: " + file.string(), false, sf::Color::Red };
    }
}
