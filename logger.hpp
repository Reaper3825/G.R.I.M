#pragma once
#include <string>
#include <chrono>

// =====================================================
// Build Mode Enum
// =====================================================
enum class BuildMode {
    Debug,
    Release
};

// Global build mode
extern BuildMode g_buildMode;

// =====================================================
// Phase Info Struct (Release mode)
// =====================================================
struct PhaseInfo {
    std::chrono::system_clock::time_point timestamp;
    std::string fileName;
    std::string phaseName;
    bool success;
};

extern PhaseInfo g_phaseInfo;

// =====================================================
// Core Logging Functions
// =====================================================
void logPhaseInternal(const std::string& file,
                      const std::string& phase,
                      bool success);

void logDebug(const std::string& tag, const std::string& msg);
void logTrace(const std::string& tag, const std::string& msg);
void logError(const std::string& tag, const std::string& msg);

// =====================================================
// Phase Group Controls
// =====================================================
void beginPhaseGroup();
void endPhaseGroup();

// =====================================================
// Logger lifecycle
// =====================================================
void initLogger(const std::string& filename = "grim.log");
void shutdownLogger();

// =====================================================
// Macros
// =====================================================
#define LOG_PHASE(phase, success) logPhaseInternal(__FILE__, phase, success)
#define LOG_DEBUG(tag, msg) logDebug(tag, msg)
#define LOG_TRACE(tag, msg) logTrace(tag, msg)
#define LOG_ERROR(tag, msg) logError(tag, msg)
