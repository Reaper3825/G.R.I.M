#include "logger.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <mutex>
#include <vector>
#include <filesystem>
#include <chrono>

// =====================================================
// Globals
// =====================================================
#if defined(_DEBUG)
BuildMode g_buildMode = BuildMode::Debug;
#else
BuildMode g_buildMode = BuildMode::Release;
#endif

PhaseInfo g_phaseInfo{};
static std::mutex g_logMutex;

// 🔹 Buffer for grouped phase logging
static bool g_buffering = false;
static std::vector<std::string> g_phaseBuffer;

// 🔹 File output stream
static std::ofstream g_logFile;

// =====================================================
// Helpers
// =====================================================
static std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static std::string nowTimestamp() {
    return formatTimestamp(std::chrono::system_clock::now());
}

static std::string basename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static void writeLine(const std::string& line) {
    // Always write to file
    if (g_logFile.is_open()) {
        g_logFile << line << std::endl;
        g_logFile.flush();
    }

    // Also write to console if it exists
    std::cerr << line << std::endl;
}

// =====================================================
// Buffering controls
// =====================================================
void beginPhaseGroup() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    g_buffering = true;
    g_phaseBuffer.clear();
}

void endPhaseGroup() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    for (auto& line : g_phaseBuffer) {
        writeLine(line);
    }
    g_phaseBuffer.clear();
    g_buffering = false;
}

// =====================================================
// Phase Logging
// =====================================================
void logPhaseInternal(const std::string& file,
                      const std::string& phase,
                      bool success)
{
    std::lock_guard<std::mutex> lock(g_logMutex);

    g_phaseInfo.timestamp = std::chrono::system_clock::now();
    g_phaseInfo.fileName  = basename(file);
    g_phaseInfo.phaseName = phase;
    g_phaseInfo.success   = success;

    std::ostringstream oss;
    oss << "| " << formatTimestamp(g_phaseInfo.timestamp)
        << " | " << g_phaseInfo.fileName
        << " | " << g_phaseInfo.phaseName
        << " | " << (g_phaseInfo.success ? "true" : "false")
        << " |";

    std::string entry = oss.str();

    if (g_buffering) {
        g_phaseBuffer.push_back(entry);
    } else {
        writeLine(entry);
    }
}

// =====================================================
// Debug / Trace / Error Logging
// =====================================================
void logDebug(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine("[" + nowTimestamp() + "][DEBUG][" + tag + "] " + msg);
}

void logTrace(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine("[" + nowTimestamp() + "][TRACE][" + tag + "] " + msg);
}

void logError(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine("[" + nowTimestamp() + "][ERROR][" + tag + "] " + msg);
}

// =====================================================
// Lifecycle
// =====================================================
namespace fs = std::filesystem;

void initLogger(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    fs::path logPath = fs::absolute(filename);
    g_logFile.open(logPath, std::ios::out | std::ios::app);

    if (g_logFile.is_open()) {
        std::string header = "==== GRIM Log Started ====";
        g_logFile << header << std::endl;

        // Always print the log path so devs/users know where to look
        std::string msg = "[" + nowTimestamp() + "][Logger] Writing logs to: " + logPath.string();
        std::cerr << msg << std::endl;    // Debug build console
        g_logFile << msg << std::endl;    // Always file
    } else {
        std::cerr << "[Logger] ERROR: Could not open log file: "
                  << logPath.string() << std::endl;
    }
}

void shutdownLogger() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile << "==== GRIM Log Ended ====" << std::endl;
        g_logFile.close();
    }
}
