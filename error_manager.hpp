#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "commands/commands_core.hpp"   // 🔹 For CommandResult

namespace Logger {
    enum class Level { DEBUG, INFO, WARN, ERROR };

    void init(const std::string& logFile);
    void log(Level level, const std::string& message);
    void logResult(const CommandResult& result);
}

class ErrorManager {
public:
    static void load(const std::string& path);
    static std::string getUserMessage(const std::string& code);
    static std::string getDebugMessage(const std::string& code);

    // 🔹 Now returns a CommandResult instead of void
    static CommandResult report(const std::string& code);

private:
    static nlohmann::json errors; // raw loaded JSON
    static nlohmann::json root;   // flattened view (either errors["errors"] or errors)
};
