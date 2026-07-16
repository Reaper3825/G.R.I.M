#include "multi_monitor.hpp"
#include "digital/DigitalCaptureSource.hpp"
#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>

#ifdef _WIN32
#include "core/grim_platform.h"
#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")
#endif

extern GRIM::MMO::HardwareInventory g_hardwareInventory;

namespace GRIM {
namespace Perception {

std::unique_ptr<MultiMonitorManager> g_multiMonitor = nullptr;

struct MultiMonitorManager::Impl {
    std::vector<ExtendedMonitorInfo> monitors;
    int totalVirtualX = 0;
    int totalVirtualY = 0;
    int totalVirtualWidth = 0;
    int totalVirtualHeight = 0;
};

MultiMonitorManager::MultiMonitorManager()
    : pImpl(std::make_unique<Impl>()) {
}

MultiMonitorManager::~MultiMonitorManager() = default;

bool MultiMonitorManager::init(const GRIM::MMO::HardwareInventory* inventory) {
    LOG_DEBUG("MultiMonitor", "Initializing multi-monitor manager");

    pImpl->monitors.clear();

    // The live OS directory is authoritative for capture. HardwareInventory is
    // a bootstrap snapshot and may be empty or stale after monitor hot-plug.
    auto source = Digital::CreatePlatformDigitalCaptureSource();
    const auto live_monitors = source ? source->EnumerateMonitors()
                                      : std::vector<Digital::DigitalMonitorDescriptor>{};
    for (const auto& mon : live_monitors) {
        ExtendedMonitorInfo extended;
        extended.x = mon.desktop_rect.x;
        extended.y = mon.desktop_rect.y;
        extended.width = mon.desktop_rect.width;
        extended.height = mon.desktop_rect.height;
        extended.is_primary = mon.is_primary;
        extended.monitorIndex = mon.index;
        extended.deviceName = mon.id;
        extended.friendlyName = mon.friendly_name;
        extended.dpiX = static_cast<int>(mon.dpi_x);
        extended.dpiY = static_cast<int>(mon.dpi_y);
        extended.scaleFactor = mon.scale_factor;
        pImpl->monitors.push_back(extended);
    }

    // Non-Windows builds do not yet have a digital source. Retain the
    // inventory projection there until their platform backend lands.
    if (pImpl->monitors.empty()) {
        const GRIM::MMO::HardwareInventory& inv = inventory ? *inventory : g_hardwareInventory;
        for (int i = 0; i < static_cast<int>(inv.monitors.size()); ++i) {
            const auto& mon = inv.monitors[i];
            ExtendedMonitorInfo extended;
            extended.x = mon.x;
            extended.y = mon.y;
            extended.width = mon.width;
            extended.height = mon.height;
            extended.is_primary = mon.is_primary;
            extended.monitorIndex = i;
            pImpl->monitors.push_back(extended);
        }
    }

    if (pImpl->monitors.empty()) {
        LOG_ERROR("MultiMonitor", "No active monitors detected by OS or hardware inventory");
        return false;
    }

    // Calculate virtual bounds (min/max of all monitor positions)
    if (!pImpl->monitors.empty()) {
        int minX = pImpl->monitors[0].x;
        int minY = pImpl->monitors[0].y;
        int maxX = pImpl->monitors[0].x + pImpl->monitors[0].width;
        int maxY = pImpl->monitors[0].y + pImpl->monitors[0].height;
        
        for (const auto& mon : pImpl->monitors) {
            minX = std::min(minX, mon.x);
            minY = std::min(minY, mon.y);
            maxX = std::max(maxX, mon.x + mon.width);
            maxY = std::max(maxY, mon.y + mon.height);
        }
        
        pImpl->totalVirtualX = minX;
        pImpl->totalVirtualY = minY;
        pImpl->totalVirtualWidth = maxX - minX;
        pImpl->totalVirtualHeight = maxY - minY;
    }
    
    // Refresh additional details from Windows API
    refreshMonitorDetails();
    
    LOG_DEBUG("MultiMonitor", "Initialized with " + std::to_string(pImpl->monitors.size()) + 
              " monitors, virtual desktop: " + std::to_string(pImpl->totalVirtualWidth) + 
              "x" + std::to_string(pImpl->totalVirtualHeight));
    
    return true;
}

void MultiMonitorManager::refreshMonitorDetails() {
#ifdef _WIN32
    // Get additional monitor info using Windows API
    for (auto& mon : pImpl->monitors) {
        // Create a rect for this monitor
        RECT monRect = {mon.x, mon.y, mon.x + mon.width, mon.y + mon.height};
        
        // Get monitor handle
        HMONITOR hMonitor = MonitorFromRect(&monRect, MONITOR_DEFAULTTONEAREST);
        
        if (hMonitor) {
            // Get monitor info
            MONITORINFOEXW monitorInfo = {};
            monitorInfo.cbSize = sizeof(MONITORINFOEXW);
            
            if (GetMonitorInfoW(hMonitor, &monitorInfo)) {
                // Get device name
                std::wstring wDeviceName(monitorInfo.szDevice);
                mon.deviceName = std::string(wDeviceName.begin(), wDeviceName.end());
                
                // Get DPI
                UINT dpiX = 96, dpiY = 96;
                if (SUCCEEDED(GetDpiForMonitor(hMonitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY))) {
                    mon.dpiX = dpiX;
                    mon.dpiY = dpiY;
                    mon.scaleFactor = static_cast<float>(dpiX) / 96.0f;
                }
                
                // Get display device info for friendly name and refresh rate
                DISPLAY_DEVICEW displayDevice = {};
                displayDevice.cb = sizeof(DISPLAY_DEVICEW);
                
                if (EnumDisplayDevicesW(monitorInfo.szDevice, 0, &displayDevice, 0)) {
                    std::wstring wFriendlyName(displayDevice.DeviceString);
                    mon.friendlyName = std::string(wFriendlyName.begin(), wFriendlyName.end());
                }
                
                // Get display settings for refresh rate
                DEVMODEW devMode = {};
                devMode.dmSize = sizeof(DEVMODEW);
                
                if (EnumDisplaySettingsW(monitorInfo.szDevice, ENUM_CURRENT_SETTINGS, &devMode)) {
                    mon.refreshRate = devMode.dmDisplayFrequency;
                    mon.bitsPerPixel = devMode.dmBitsPerPel;
                    mon.orientation = devMode.dmDisplayOrientation;
                }
            }
        }
    }
#endif
}

std::vector<ExtendedMonitorInfo> MultiMonitorManager::getMonitors() const {
    return pImpl->monitors;
}

ExtendedMonitorInfo MultiMonitorManager::getPrimaryMonitor() const {
    for (const auto& mon : pImpl->monitors) {
        if (mon.is_primary) {
            return mon;
        }
    }
    
    // Fallback to first monitor
    if (!pImpl->monitors.empty()) {
        return pImpl->monitors[0];
    }
    
    return ExtendedMonitorInfo();
}

ExtendedMonitorInfo MultiMonitorManager::getMonitorByIndex(int index) const {
    if (index >= 0 && index < static_cast<int>(pImpl->monitors.size())) {
        return pImpl->monitors[index];
    }
    return ExtendedMonitorInfo();
}

ExtendedMonitorInfo MultiMonitorManager::getMonitorContainingPoint(int x, int y) const {
    for (const auto& mon : pImpl->monitors) {
        if (x >= mon.x && x < mon.x + mon.width &&
            y >= mon.y && y < mon.y + mon.height) {
            return mon;
        }
    }
    
    // Return primary if point not in any monitor
    return getPrimaryMonitor();
}

void MultiMonitorManager::getVirtualScreenBounds(int& x, int& y, int& width, int& height) const {
    x = pImpl->totalVirtualX;
    y = pImpl->totalVirtualY;
    width = pImpl->totalVirtualWidth;
    height = pImpl->totalVirtualHeight;
}

cv::Mat MultiMonitorManager::captureMonitor(int monitorIndex) const {
    if (monitorIndex < 0 || monitorIndex >= static_cast<int>(pImpl->monitors.size())) {
        LOG_ERROR("MultiMonitor", "Invalid monitor index: " + std::to_string(monitorIndex));
        return cv::Mat();
    }
    
    auto source = Digital::CreatePlatformDigitalCaptureSource();
    if (!source) return {};
    Digital::DigitalCaptureRequest request;
    request.mode = Digital::DigitalCaptureMode::Monitor;
    request.monitor_index = monitorIndex;
    auto result = source->Capture(request);
    if (!result.Succeeded()) {
        LOG_ERROR("MultiMonitor", "Capture monitor failed: " + result.metadata.error);
        return {};
    }
    return result.image;
}

cv::Mat MultiMonitorManager::capturePrimaryMonitor() const {
    for (size_t i = 0; i < pImpl->monitors.size(); ++i) {
        if (pImpl->monitors[i].is_primary) {
            return captureMonitor(static_cast<int>(i));
        }
    }
    
    // Fallback to first monitor
    if (!pImpl->monitors.empty()) {
        return captureMonitor(0);
    }
    
    return cv::Mat();
}

cv::Mat MultiMonitorManager::captureAllMonitors() const {
    auto source = Digital::CreatePlatformDigitalCaptureSource();
    if (!source) return {};
    Digital::DigitalCaptureRequest request;
    request.mode = Digital::DigitalCaptureMode::VirtualDesktop;
    auto result = source->Capture(request);
    if (!result.Succeeded()) {
        LOG_ERROR("MultiMonitor", "Capture virtual desktop failed: " + result.metadata.error);
        return {};
    }
    return result.image;
}

cv::Mat MultiMonitorManager::captureActiveMonitor() const {
    auto source = Digital::CreatePlatformDigitalCaptureSource();
    if (!source) return {};
    Digital::DigitalCaptureRequest request;
    request.mode = Digital::DigitalCaptureMode::ActiveMonitor;
    auto result = source->Capture(request);
    if (!result.Succeeded()) {
        LOG_ERROR("MultiMonitor", "Capture active monitor failed: " + result.metadata.error);
        return {};
    }
    return result.image;
}

int MultiMonitorManager::getMonitorCount() const {
    return static_cast<int>(pImpl->monitors.size());
}

bool MultiMonitorManager::isMultiMonitorSetup() const {
    return pImpl->monitors.size() > 1;
}

int MultiMonitorManager::getActiveMonitorIndex() const {
#ifdef _WIN32
    HWND hwnd = GetForegroundWindow();
    if (!hwnd) {
        return getPrimaryMonitor().monitorIndex;
    }
    
    RECT rect;
    if (!GetWindowRect(hwnd, &rect)) {
        return getPrimaryMonitor().monitorIndex;
    }
    
    int centerX = (rect.left + rect.right) / 2;
    int centerY = (rect.top + rect.bottom) / 2;
    
    return getMonitorContainingPoint(centerX, centerY).monitorIndex;
#else
    return 0;
#endif
}

std::string MultiMonitorManager::getMonitorSummary() const {
    std::ostringstream ss;
    
    ss << "=== Multi-Monitor Configuration ===\n";
    ss << "Total monitors: " << pImpl->monitors.size() << "\n";
    ss << "Virtual desktop: " << pImpl->totalVirtualWidth << "x" << pImpl->totalVirtualHeight << "\n\n";
    
    for (const auto& mon : pImpl->monitors) {
        ss << "Monitor " << mon.monitorIndex << (mon.is_primary ? " [PRIMARY]" : "") << ":\n";
        
        if (!mon.friendlyName.empty()) {
            ss << "  Name: " << mon.friendlyName << "\n";
        }
        if (!mon.deviceName.empty()) {
            ss << "  Device: " << mon.deviceName << "\n";
        }
        
        ss << "  Resolution: " << mon.width << "x" << mon.height << "\n";
        ss << "  Position: (" << mon.x << ", " << mon.y << ")\n";
        ss << "  DPI: " << mon.dpiX << "x" << mon.dpiY << " (scale: " << mon.scaleFactor << "x)\n";
        ss << "  Refresh: " << mon.refreshRate << " Hz\n";
        ss << "  Depth: " << mon.bitsPerPixel << " bits\n\n";
    }
    
    return ss.str();
}

// Global initialization
void initMultiMonitor() {
    if (!g_multiMonitor) {
        g_multiMonitor = std::make_unique<MultiMonitorManager>();
        g_multiMonitor->init();
    }
}

int getMonitorCount() {
    if (!g_multiMonitor) {
        initMultiMonitor();
    }
    return g_multiMonitor ? g_multiMonitor->getMonitorCount() : 1;
}

cv::Mat captureActiveMonitor() {
    if (!g_multiMonitor) {
        initMultiMonitor();
    }
    return g_multiMonitor ? g_multiMonitor->captureActiveMonitor() : cv::Mat();
}

ExtendedMonitorInfo getActiveMonitorInfo() {
    if (!g_multiMonitor) {
        initMultiMonitor();
    }
    
    if (g_multiMonitor) {
        int activeIndex = g_multiMonitor->getActiveMonitorIndex();
        return g_multiMonitor->getMonitorByIndex(activeIndex);
    }
    
    return ExtendedMonitorInfo();
}

} // namespace Perception
} // namespace GRIM
