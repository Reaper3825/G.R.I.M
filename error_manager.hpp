#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>

// Forward declaration to avoid circular include
struct CommandResult;
class ConsoleHistory;

// -------------------------
// Logger
// -------------------------
namespace Logger {
    enum class Level { DEBUG, INFO, WARN, ERROR };

    void init(const std::string& logFile);
    void log(Level level, const std::string& message);
    void logResult(const CommandResult& result);
}

// -------------------------
// ErrorManager
// -------------------------
class ErrorManager {
public:
    static nlohmann::json errors;

    static void load(const std::string& path);
    static std::string getUserMessage(const std::string& code);
    static std::string getDebugMessage(const std::string& code);
    static void report(const std::string& code);
};
