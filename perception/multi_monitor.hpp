#pragma once
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "system_detect.hpp" // Use existing monitor detection

#ifdef _WIN32
#include <windows.h>
#endif

namespace GRIM {
namespace Perception {

// Extended monitor info with perception-specific details
struct ExtendedMonitorInfo : public ::MonitorInfo {
    int monitorIndex;          // 0-based index
    std::string deviceName;    // e.g., "\\\\.\\DISPLAY1"
    std::string friendlyName;  // e.g., "Dell U2719D"
    
    // Additional characteristics
    int dpiX = 96, dpiY = 96;
    float scaleFactor = 1.0f;  // e.g., 1.0, 1.25, 1.5, 2.0
    int refreshRate = 60;      // Hz
    int bitsPerPixel = 32;
    int orientation = 0;       // 0=landscape, 90=portrait, etc.
};

// Multi-monitor manager
class MultiMonitorManager {
public:
    MultiMonitorManager();
    ~MultiMonitorManager();
    
    // Initialize using system detection data
    bool init(const SystemInfo* sysInfo = nullptr);
    
    // Get all detected monitors (from system_detect)
    std::vector<ExtendedMonitorInfo> getMonitors() const;
    
    // Get specific monitor
    ExtendedMonitorInfo getPrimaryMonitor() const;
    ExtendedMonitorInfo getMonitorByIndex(int index) const;
    ExtendedMonitorInfo getMonitorContainingPoint(int x, int y) const;
    
    // Get total virtual screen bounds (all monitors combined)
    void getVirtualScreenBounds(int& x, int& y, int& width, int& height) const;
    
    // Capture operations
    cv::Mat captureMonitor(int monitorIndex) const;
    cv::Mat capturePrimaryMonitor() const;
    cv::Mat captureAllMonitors() const; // Entire virtual desktop
    cv::Mat captureActiveMonitor() const; // Monitor with active window
    
    // Helper queries
    int getMonitorCount() const;
    bool isMultiMonitorSetup() const;
    int getActiveMonitorIndex() const; // Monitor with foreground window
    
    // Monitor information
    std::string getMonitorSummary() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Refresh monitor info from Windows API
    void refreshMonitorDetails();
};

// Global multi-monitor manager
extern std::unique_ptr<MultiMonitorManager> g_multiMonitor;

// Initialize global multi-monitor manager
void initMultiMonitor();

// Quick access functions
int getMonitorCount();
cv::Mat captureActiveMonitor();
ExtendedMonitorInfo getActiveMonitorInfo();

} // namespace Perception
} // namespace GRIM
