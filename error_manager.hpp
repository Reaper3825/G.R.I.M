#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "commands/commands_core.hpp"

// ====================================================
// BGFX-Compatible Color Defines
// ====================================================
#define COLOR_RGBA(r,g,b,a) \
    ((uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r))

#define COLOR_RED    COLOR_RGBA(255,0,0,255)
#define COLOR_GREEN  COLOR_RGBA(0,255,0,255)
#define COLOR_YELLOW COLOR_RGBA(255,255,0,255)
#define COLOR_WHITE  COLOR_RGBA(255,255,255,255)

// ====================================================
// Logger
// ====================================================
namespace Logger {

    enum class Level {
        Debug,
        Info,
        Warn,
        Error
    };

    void init(const std::string& logFile);
    void logMessage(Level level, const std::string& message);
    void logResult(const CommandResult& result);
}

// ====================================================
// ErrorManager
// ====================================================
class ErrorManager {
public:
    static void load(const std::string& path);
    static std::string getUserMessage(const std::string& code);
    static std::string getDebugMessage(const std::string& code);
    static CommandResult report(const std::string& code);

private:
    static nlohmann::json errors;
    static nlohmann::json root;
};

