#include "commands_filesystem.hpp"
#include "console_history.hpp"
#include <filesystem>
#include <iostream>
#include <SFML/Graphics.hpp>

extern ConsoleHistory history;
extern std::filesystem::path g_currentDir;

// ------------------------------------------------------------
// pwd → show current directory
// ------------------------------------------------------------
CommandResult cmdShowPwd([[maybe_unused]] const std::string& arg) {
    return {
        "[FS] Current directory: " + g_currentDir.string(),
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Current directory shown",
        "summary"
    };
}

// ------------------------------------------------------------
// cd <dir> → change directory
// ------------------------------------------------------------
CommandResult cmdChangeDir(const std::string& arg) {
    if (arg.empty()) {
        return {
            "[FS] Usage: cd <directory>",
            false,
            sf::Color::Red,
            "ERR_FS_NO_ARGUMENT",
            "Directory name required",
            "error"
        };
    }

    std::filesystem::path newPath = g_currentDir / arg;
    if (!std::filesystem::exists(newPath)) {
        return {
            "[FS] Directory does not exist: " + arg,
            false,
            sf::Color::Red,
            "ERR_FS_NOT_FOUND",
            "Directory not found",
            "error"
        };
    }

    g_currentDir = std::filesystem::canonical(newPath);
    return {
        "[FS] Changed directory to: " + g_currentDir.string(),
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Directory changed",
        "routine"
    };
}

// ------------------------------------------------------------
// ls → list directory contents
// ------------------------------------------------------------
CommandResult cmdListDir([[maybe_unused]] const std::string& arg) {
    std::string output = "[FS] Contents:\n";
    for (const auto& entry : std::filesystem::directory_iterator(g_currentDir)) {
        output += " - " + entry.path().filename().string() + "\n";
    }

    return {
        output,
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Directory contents listed",
        "summary"
    };
}

// ------------------------------------------------------------
// mkdir <dir> → make directory
// ------------------------------------------------------------
CommandResult cmdMakeDir(const std::string& arg) {
    if (arg.empty()) {
        return {
            "[FS] Usage: mkdir <directory>",
            false,
            sf::Color::Red,
            "ERR_FS_NO_ARGUMENT",
            "Directory name required",
            "error"
        };
    }

    std::filesystem::path newDir = g_currentDir / arg;
    if (std::filesystem::create_directory(newDir)) {
        return {
            "[FS] Directory created: " + newDir.string(),
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Directory created",
            "routine"
        };
    } else {
        return {
            "[FS] Failed to create directory: " + newDir.string(),
            false,
            sf::Color::Red,
            "ERR_FS_CREATE_FAILED",
            "Failed to create directory",
            "error"
        };
    }
}

// ------------------------------------------------------------
// rm <file> → remove file
// ------------------------------------------------------------
CommandResult cmdRemoveFile(const std::string& arg) {
    if (arg.empty()) {
        return {
            "[FS] Usage: rm <file>",
            false,
            sf::Color::Red,
            "ERR_FS_NO_ARGUMENT",
            "File name required",
            "error"
        };
    }

    std::filesystem::path file = g_currentDir / arg;
    if (!std::filesystem::exists(file)) {
        return {
            "[FS] File not found: " + arg,
            false,
            sf::Color::Red,
            "ERR_FS_NOT_FOUND",
            "File not found",
            "error"
        };
    }

    if (std::filesystem::remove(file)) {
        return {
            "[FS] Removed: " + file.string(),
            true,
            sf::Color::Green,
            "ERR_NONE",
            "File removed",
            "routine"
        };
    } else {
        return {
            "[FS] Failed to remove: " + file.string(),
            false,
            sf::Color::Red,
            "ERR_FS_REMOVE_FAILED",
            "Failed to remove file",
            "error"
        };
    }
}
