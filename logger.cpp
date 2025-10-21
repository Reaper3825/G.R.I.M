#include "pch.hpp"
#include "logger.hpp"
#include <iomanip>
#include <sstream>
#include <iostream>

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
static bool g_buffering = false;
static std::vector<std::string> g_phaseBuffer;
static std::ofstream g_logFile;
static std::ofstream g_rewardFile; // separate reward log
static RewardStats g_rewardStats{};

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

static void writeLine(std::ofstream& stream, const std::string& line) {
    if (stream.is_open()) {
        stream << line << std::endl;
        stream.flush();
    }
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
        writeLine(g_logFile, line);
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
        writeLine(g_logFile, entry);
    }
}

// =====================================================
// Debug / Trace / Error Logging
// =====================================================
void logDebug(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine(g_logFile, "[" + nowTimestamp() + "][DEBUG][" + tag + "] " + msg);
}

void logTrace(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine(g_logFile, "[" + nowTimestamp() + "][TRACE][" + tag + "] " + msg);
}

void logError(const std::string& tag, const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    writeLine(g_logFile, "[" + nowTimestamp() + "][ERROR][" + tag + "] " + msg);
}

// =====================================================
// Reward Logging
// =====================================================
void logReward(float base, float time, float sentiment, float category, float diversity, float total) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    // Update rolling stats
    g_rewardStats.count++;
    g_rewardStats.lastReward = total;
    g_rewardStats.rollingMean += (total - g_rewardStats.rollingMean) / static_cast<float>(g_rewardStats.count);

    std::ostringstream oss;
    oss << "[" << nowTimestamp() << "][REWARD] "
        << "Base: " << std::fixed << std::setprecision(2) << base
        << " | Time: " << time
        << " | Sent: " << sentiment
        << " | Cat: " << category
        << " | Div: " << diversity
        << " | Total: " << total
        << " | Mean: " << g_rewardStats.rollingMean
        << " (" << g_rewardStats.count << " samples)";

    writeLine(g_logFile, oss.str());
    writeLine(g_rewardFile, oss.str()); // separate file sink
}

// =====================================================
// Lifecycle
// =====================================================
namespace fs = std::filesystem;

void initLogger(const std::string& filename) {
    std::lock_guard<std::mutex> lock(g_logMutex);

    fs::path logPath = fs::absolute(filename);
    fs::path rewardPath = logPath.parent_path() / "reward.log";

    g_logFile.open(logPath, std::ios::out | std::ios::trunc);
    g_rewardFile.open(rewardPath, std::ios::out | std::ios::trunc);

    if (g_logFile.is_open()) {
        std::string header = "==== GRIM Log Started ====";
        g_logFile << header << std::endl;

        std::string msg = "[" + nowTimestamp() + "][Logger] Writing logs to: " + logPath.string();
        std::cerr << msg << std::endl;
        g_logFile << msg << std::endl;
    } else {
        std::cerr << "[Logger] ERROR: Could not open log file: " << logPath.string() << std::endl;
    }

    if (g_rewardFile.is_open()) {
        g_rewardFile << "==== Reward Log Started ====" << std::endl;
    } else {
        std::cerr << "[Logger] ERROR: Could not open reward.log file." << std::endl;
    }
}

void shutdownLogger() {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (g_logFile.is_open()) {
        g_logFile << "==== GRIM Log Ended ====" << std::endl;
        g_logFile.close();
    }
    if (g_rewardFile.is_open()) {
        g_rewardFile << "==== Reward Log Ended ====" << std::endl;
        g_rewardFile.close();
    }
}
