#pragma once
#include <string>
#include <vector>

//wifi check
extern bool g_wifiConnected;

// =========================================================
// Geolocation information structure
// =========================================================
struct LocationInfo {
    std::string ip;
    std::string country;
    std::string region;
    std::string city;
    std::string postal;
    double lat = 0.0;
    double lon = 0.0;
    std::string isp;

    std::string shortAddress() const {
        if (!city.empty() && !region.empty()) return city + ", " + region;
        if (!city.empty()) return city;
        if (!region.empty()) return region;
        return country.empty() ? "Unknown" : country;
    }

    std::string fullAddress() const {
        std::string out;
        if (!city.empty()) out += city;
        if (!region.empty()) { if (!out.empty()) out += ", "; out += region; }
        if (!postal.empty()) { if (!out.empty()) out += " "; out += postal; }
        if (!country.empty()) { if (!out.empty()) out += ", "; out += country; }
        return out.empty() ? "Unknown location" : out;
    }
};

extern LocationInfo g_location;
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
    int virtualOriginX = 0;     // top-left virtual desktop origin
    int virtualOriginY = 0;
    std::vector<MonitorInfo> monitors; // per-monitor info
};

// =========================================================
// Functions
// =========================================================
SystemInfo detectSystem();
void logSystemInfo(const SystemInfo& info);
std::string chooseWhisperModel(const SystemInfo& info);
