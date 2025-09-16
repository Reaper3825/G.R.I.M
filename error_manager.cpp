#include "error_manager.hpp"
#include "commands/commands_core.hpp" // For CommandResult
#include "console_history.hpp"

#include <fstream>
#include <iostream>
#include <ctime>
#include <filesystem>

static std::ofstream logStream;

namespace Logger {

    void init(const std::string& logFile) {
        logStream.open(logFile, std::ios::app);
        if (!logStream.is_open()) {
            std::cerr << "[Logger] Failed to open log file: " << logFile << std::endl;
        }
    }

    static std::string timestamp() {
        std::time_t t = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        return std::string(buf);
    }

    static std::string levelToString(Level lvl) {
        switch (lvl) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO:  return "INFO";
            case Level::WARN:  return "WARN";
            case Level::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }

    void log(Level level, const std::string& message) {
        std::string line = "[" + timestamp() + "][" + levelToString(level) + "] " + message;

        if (level == Level::WARN || level == Level::ERROR)
            std::cerr << line << std::endl;
        else
            std::cout << line << std::endl;

        if (logStream.is_open()) {
            logStream << line << std::endl;
        }
    }

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
nlohmann::json ErrorManager::root;

void ErrorManager::load(const std::string& path) {
    namespace fs = std::filesystem;

    std::ifstream in(path);
    if (!in) {
        std::cerr << "[ErrorManager] Could not open " << path << std::endl;
        return;
    }

    try {
        in >> errors;
        std::cout << "[ErrorManager] Loaded errors.json from: " 
                  << fs::absolute(path).string() << std::endl;

        // Handle nested "errors" key if present
        if (errors.contains("errors") && errors["errors"].is_object()) {
            root = errors["errors"];
        } else {
            root = errors;
        }

        // Dump loaded keys
        std::cout << "[ErrorManager] Available error codes: ";
        for (auto& [key, val] : root.items()) {
            std::cout << key << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ErrorManager] Failed to parse " << path 
                  << " -> " << e.what() << std::endl;
    }
}

std::string ErrorManager::getUserMessage(const std::string& code) {
    if (root.contains(code) && root[code].contains("user")) {
        return root[code]["user"];
    }
    return "[Error] Unknown error code: " + code;
}

std::string ErrorManager::getDebugMessage(const std::string& code) {
    if (root.contains(code) && root[code].contains("debug")) {
        return root[code]["debug"];
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
