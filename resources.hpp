#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include "console_history.hpp"
#include "timer.hpp"

// ====================================================
// Constants
// ====================================================
inline constexpr const char* AI_CONFIG_FILE = "ai_config.json";

// ====================================================
// Resource loading
// ====================================================
std::string getResourcePath();
std::string loadTextResource(const std::string& filename, int argc, char** argv);
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history = nullptr);

// ====================================================
// Global memory + AI config (JSON containers only)
// ====================================================
extern nlohmann::json longTermMemory;
extern nlohmann::json aiConfig;

// ====================================================
// Global runtime state
// ====================================================
extern ConsoleHistory history;             // Console + speech log
extern std::vector<Timer> timers;          // Active timers
extern std::filesystem::path g_currentDir; // Filesystem current working dir

// ====================================================
// Memory helpers
// ====================================================
void loadMemory();
void saveMemory();
void rememberCorrection(const std::string& wrong, const std::string& right);
void rememberShortcut(const std::string& phrase, const std::string& command);
void getSafeResourcePath(std::string& outPath);
void incrementUsageCount(const std::string& command);


// Path resolution modes
enum class PathResolutionMode {
    Absolute,  // Treat as absolute path
    Relative,  // Resolve relative to GRIM root
    Search     // Search for file in GRIM directory tree
};

std::string getSafeResourcePath(const std::string& target, PathResolutionMode mode = PathResolutionMode::Relative);
std::vector<std::string> listFiles(const std::string& folderPath);
// ====================================================
// Global logging helper (system-level, not user history)
// ====================================================
void grimLog(const std::string& msg);

