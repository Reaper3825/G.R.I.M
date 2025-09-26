#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <SFML/Graphics/Color.hpp>

// Undefine Windows ERROR macro if it leaks in
#ifdef ERROR
#undef ERROR
#endif

#include "commands/commands_core.hpp"

// ------------------------------------------------------------
// Logger
// ------------------------------------------------------------
namespace Logger {
    enum class Level { Debug, Info, Warn, Error };

    void init(const std::string& logFile);
    void logMessage(Level level, const std::string& message);
    void logResult(const CommandResult& result);
}

// ------------------------------------------------------------
// ErrorManager
// ------------------------------------------------------------
namespace ErrorManager {
    // Load error codes from JSON (errors.json)
    void load(const std::string& path);

    // Get messages
    std::string getUserMessage(const std::string& code);
    std::string getDebugMessage(const std::string& code);

    // Report an error (returns CommandResult)
    CommandResult report(const std::string& code);

    // Internal storage
    extern nlohmann::json errors;
    extern nlohmann::json root;
}
