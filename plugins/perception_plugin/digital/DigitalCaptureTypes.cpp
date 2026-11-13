#include "DigitalCaptureTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

const char* ToString(DigitalCaptureMode mode) noexcept {
    switch (mode) {
        case DigitalCaptureMode::ActiveMonitor:  return "active-monitor";
        case DigitalCaptureMode::Monitor:        return "monitor";
        case DigitalCaptureMode::ActiveWindow:  return "active-window";
        case DigitalCaptureMode::VirtualDesktop: return "virtual-desktop";
    }
    return "unknown";
}

const char* ToString(DigitalCaptureStatus status) noexcept {
    switch (status) {
        case DigitalCaptureStatus::Ok:                return "ok";
        case DigitalCaptureStatus::Unsupported:       return "unsupported";
        case DigitalCaptureStatus::NoDisplays:        return "no-displays";
        case DigitalCaptureStatus::InvalidRequest:    return "invalid-request";
        case DigitalCaptureStatus::SourceUnavailable: return "source-unavailable";
        case DigitalCaptureStatus::PermissionDenied:  return "permission-denied";
        case DigitalCaptureStatus::CaptureFailed:     return "capture-failed";
    }
    return "unknown";
}

}}} // namespace GRIM::Perception::Digital
