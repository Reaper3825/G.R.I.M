#include "error_manager.hpp"
#include "commands/commands_core.hpp" // ✅ Needed for CommandResult definition
#include "console_history.hpp"
#include <fstream>
#include <iostream>
#include <ctime>

static std::ofstream logStream;

namespace Logger {

    // -------------------------
    // Initialize log file
    // -------------------------
    void init(const std::string& logFile) {
        logStream.open(logFile, std::ios::app);
        if (!logStream.is_open()) {
            std::cerr << "[Logger] Failed to open log file: " << logFile << std::endl;
        }
    }

    // -------------------------
    // Timestamp utility
    // -------------------------
    static std::string timestamp() {
        std::time_t t = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        return std::string(buf);
    }

    // -------------------------
    // Convert Level -> string
    // -------------------------
    static std::string levelToString(Level lvl) {
        switch (lvl) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return "INFO";
            case Level::WARN:  return "WARN";
            case Level::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }

    // -------------------------
    // Log a message
    // -------------------------
    void log(Level level, const std::string& message) {
        std::string line = "[" + timestamp() + "][" + levelToString(level) + "] " + message;

        // Console output
        if (level == Level::WARN || level == Level::ERROR)
            std::cerr << line << std::endl;
        else
            std::cout << line << std::endl;

        // File output
        if (logStream.is_open()) {
            logStream << line << std::endl;
        }
    }

    // -------------------------
    // Log CommandResult
    // -------------------------
    void logResult(const CommandResult& result) {
        if (result.success) {
            log(Level::INFO, result.message);
        } else {
            if (!result.errorCode.empty()) {
                std::string debugMsg = ErrorManager::getDebugMessage(result.errorCode);
                log(Level::ERROR, result.errorCode + " -> " + debugMsg);
            } else {
                log(Level::ERROR, result.message);
            }
        }
    }

} // namespace Logger

// -------------------------
// ErrorManager Implementation
// -------------------------
nlohmann::json ErrorManager::errors;

void ErrorManager::load(const std::string& path) {
    std::ifstream in(path);
    if (in) {
        in >> errors;
    } else {
        std::cerr << "[ErrorManager] Could not open " << path << std::endl;
    }
}

std::string ErrorManager::getUserMessage(const std::string& code) {
    if (errors.contains(code) && errors[code].contains("user")) {
        return errors[code]["user"];
    }
    return "[Error] Unknown error code: " + code;
}

std::string ErrorManager::getDebugMessage(const std::string& code) {
    if (errors.contains(code) && errors[code].contains("debug")) {
        return errors[code]["debug"];
    }
    return "[Debug] No debug message for code: " + code;
}

void ErrorManager::report(const std::string& code) {
    std::string userMsg  = getUserMessage(code);
    std::string debugMsg = getDebugMessage(code);

    extern ConsoleHistory history;
    history.push(userMsg, sf::Color::Red);

    Logger::log(Logger::Level::ERROR, code + " -> " + debugMsg);
}
