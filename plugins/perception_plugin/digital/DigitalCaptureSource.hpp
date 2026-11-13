#pragma once

#include <memory>
#include <string>
#include <vector>

#include "DigitalCaptureTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

class DigitalCaptureSource {
public:
    virtual ~DigitalCaptureSource() = default;

    virtual std::string BackendName() const = 0;
    virtual std::vector<DigitalMonitorDescriptor> EnumerateMonitors() = 0;
    virtual DigitalCaptureResult Capture(const DigitalCaptureRequest& request) = 0;
};

std::unique_ptr<DigitalCaptureSource> CreatePlatformDigitalCaptureSource();

// Must be called before creating windows. On Windows this opts into per-monitor
// DPI-aware coordinates so monitor rectangles and captured pixels use the same
// physical-pixel coordinate space.
bool EnsureDigitalCaptureDpiAwareness(std::string* error = nullptr);

// Excludes a GRIM-owned top-level window from OS capture where the platform
// supports it. `native_window` is HWND on Windows and ignored elsewhere.
bool SetDigitalCaptureExcludedWindow(void* native_window, std::string* error = nullptr);

}}} // namespace GRIM::Perception::Digital
