#pragma once
#include <mutex>
#include <chrono>

#ifdef _WIN32
    #include "core/grim_platform.h"
#endif

// =========================================================
// Resource usage structure
// =========================================================
struct ResourceUsage {
    float cpuUsage = 0.0f;      // CPU usage percentage (0-100)
    float memoryUsage = 0.0f;   // Memory usage percentage (0-100)
    float gpuUsage = 0.0f;      // GPU usage percentage (0-100)
    
    // Additional detailed info
    size_t totalMemoryMB = 0;
    size_t usedMemoryMB = 0;
    size_t totalGpuMemoryMB = 0;
    size_t usedGpuMemoryMB = 0;
};

// =========================================================
// Resource Monitor - Singleton for system-wide resource tracking
// =========================================================
class ResourceMonitor {
public:
    // Get singleton instance
    static ResourceMonitor& getInstance();
    
    // Delete copy/move constructors for singleton
    ResourceMonitor(const ResourceMonitor&) = delete;
    ResourceMonitor& operator=(const ResourceMonitor&) = delete;
    ResourceMonitor(ResourceMonitor&&) = delete;
    ResourceMonitor& operator=(ResourceMonitor&&) = delete;
    
    // Initialize the monitor (called once at startup)
    void initialize();
    
    // Update resource readings (call periodically, e.g., every 0.5s)
    void update();
    
    // Get current resource usage
    ResourceUsage getCurrentUsage() const;
    
    // Get individual metrics
    float getCpuUsage() const;
    float getMemoryUsage() const;
    float getGpuUsage() const;
    
    // Get detailed memory info
    size_t getTotalMemoryMB() const;
    size_t getUsedMemoryMB() const;
    size_t getTotalGpuMemoryMB() const;
    size_t getUsedGpuMemoryMB() const;
    
    // Check if GPU monitoring is available
    bool hasGpuMonitoring() const;
    
private:
    ResourceMonitor();
    ~ResourceMonitor() = default;
    
    // Platform-specific sampling methods
    void sampleCpuUsage();
    void sampleMemoryUsage();
    void sampleGpuUsage();
    
    // Current resource usage
    ResourceUsage currentUsage;
    mutable std::mutex usageMutex;
    
    // Initialization flag
    bool initialized;
    bool gpuAvailable;
    
#ifdef _WIN32
    // Windows-specific CPU tracking
    ULARGE_INTEGER lastCPU;
    ULARGE_INTEGER lastSysCPU;
    ULARGE_INTEGER lastUserCPU;
    int numProcessors;
    bool cpuInitialized;
#endif
    
    // Timing for update throttling
    std::chrono::steady_clock::time_point lastUpdateTime;
    double minUpdateInterval; // Minimum seconds between updates
};
