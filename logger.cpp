#include "logger.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <vector>

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

static std::string basename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
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
        std::cerr << line << std::endl;
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
    if (g_buildMode == BuildMode::Release) {
        oss << "| " << formatTimestamp(g_phaseInfo.timestamp)
            << " | " << g_phaseInfo.fileName
            << " | " << g_phaseInfo.phaseName
            << " | " << (g_phaseInfo.success ? "true" : "false")
            << " |";
    } else {
        oss << "[DEBUG][PHASE] (" << g_phaseInfo.fileName << ") "
            << g_phaseInfo.phaseName << " -> "
            << (g_phaseInfo.success ? "OK" : "FAIL");
    }

    std::string entry = oss.str();

    if (g_buffering) {
        g_phaseBuffer.push_back(entry);
    } else {
        std::cerr << entry << std::endl;
    }
}

// =====================================================
// Debug / Trace / Error Logging
// =====================================================
void logDebug(const std::string& tag, const std::string& msg) {
    if (g_buildMode == BuildMode::Debug) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        std::cerr << "[DEBUG][" << tag << "] " << msg
                  << std::endl << std::flush;
    }
}

void logTrace(const std::string& tag, const std::string& msg) {
    if (g_buildMode == BuildMode::Debug) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        std::cerr << "[TRACE][" << tag << "] " << msg
                  << std::endl << std::flush;
    }
}

void logError(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::cerr << "[ERROR][" << tag << "] " << msg
              << std::endl << std::flush;
}
