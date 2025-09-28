#pragma once
#include <string>
#include <vector>

// =========================================================
// Per-monitor information
// =========================================================
struct MonitorInfo {
    int x = 0;        // top-left corner X
    int y = 0;        // top-left corner Y
    int width = 0;
    int height = 0;
    bool isPrimary = false;
};

// =========================================================
// System information structure
// =========================================================
struct SystemInfo {
    std::string osName;
    std::string arch;

    int cpuCores = 0;
    long ramMB = 0;

    bool hasGPU = false;
    bool hasCUDA = false;
    bool hasMetal = false;
    bool hasROCm = false;
    int gpuCount = 0;
    std::string gpuName;
    long gpuVRAM_MB = 0;
    std::string gpuDriver;

    bool hasSAPI = false;
    bool hasSay = false;
    bool hasPiper = false;

    std::string outputDevice;
    std::string suggestedModel;

    // Monitor info
    bool hasMonitor = false;
    int monitorCount = 0;
    int totalScreenWidth = 0;   // full virtual desktop width
    int totalScreenHeight = 0;  // full virtual desktop height
    std::vector<MonitorInfo> monitors; // per-monitor info
};

// =========================================================
// Functions
// =========================================================
SystemInfo detectSystem();
void logSystemInfo(const SystemInfo& info);
std::string chooseWhisperModel(const SystemInfo& info);
