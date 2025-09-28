#pragma once
#include <string>
#include <atomic>
#include <chrono>

namespace Wake {

// Type of stimulant
enum class Stimulant {
    Unknown,
    Voice,
    Motion,
    Alarm,
    Keypress
};

// Event struct
struct WakeEvent {
    Stimulant stimulant;
    std::string source;
    float intensity;
    int priority;
    std::chrono::steady_clock::time_point timestamp;
    std::string payload;
};

// Global awake state
extern std::atomic<bool> g_awake;

// Central API
void init();
void shutdown();
void triggerWake(const WakeEvent& ev);

} // namespace Wake
