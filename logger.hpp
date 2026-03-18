#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <fstream>
#include <filesystem>

// Export macros for logger functions
#if defined(_WIN32)
  #if defined(GRIM_BUILD_HOST)
    #define GRIM_LOGGER_API __declspec(dllexport)
  #else
    #define GRIM_LOGGER_API __declspec(dllimport)
  #endif
#else
  #define GRIM_LOGGER_API __attribute__((visibility("default")))
#endif

// =====================================================
// Build Info
// =====================================================
enum class BuildMode {
    Debug,
    Release
};

struct PhaseInfo {
    std::chrono::system_clock::time_point timestamp;
    std::string fileName;
    std::string phaseName;
    bool success;
};

// =====================================================
// Logger Interface
// =====================================================
extern BuildMode g_buildMode;
extern PhaseInfo g_phaseInfo;

// Core functions
void initLogger(const std::string& filename);
void shutdownLogger();
void beginPhaseGroup();
void endPhaseGroup();
void logPhaseInternal(const std::string& file, const std::string& phase, bool success);

// =====================================================
// Log Levels
// =====================================================
enum class LogLevel {
    Phase,
    Warning,
    Error,
    Debug,
    Reward
};

// =====================================================
// Reward Tracker
// =====================================================
struct RewardStats {
    float rollingMean = 0.0f;
    float lastReward  = 0.0f;
    size_t count      = 0;
};

GRIM_LOGGER_API void logDebug(const std::string& tag, const std::string& msg);
GRIM_LOGGER_API void logTrace(const std::string& tag, const std::string& msg);
GRIM_LOGGER_API void logError(const std::string& tag, const std::string& msg);
GRIM_LOGGER_API void logReward(float base, float time, float sentiment, float category, float diversity, float total);

// =====================================================
// Macros for ease of use
// =====================================================
#define LOG_DEBUG(tag, msg)   logDebug(tag, msg)
#define LOG_TRACE(tag, msg)   logTrace(tag, msg)
#define LOG_ERROR(tag, msg)   logError(tag, msg)
#define LOG_PHASE(phase, success) logPhaseInternal(__FILE__, phase, success)
#define LOG_REWARD(b,t,s,c,d,tot) logReward(b,t,s,c,d,tot)
