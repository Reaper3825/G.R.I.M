#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Digital {

enum class DigitalCaptureMode : std::uint8_t {
    ActiveMonitor = 0,
    Monitor,
    ActiveWindow,
    VirtualDesktop
};

enum class DigitalCaptureStatus : std::uint8_t {
    Ok = 0,
    Unsupported,
    NoDisplays,
    InvalidRequest,
    SourceUnavailable,
    PermissionDenied,
    CaptureFailed
};

struct DigitalRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    bool IsValid() const noexcept { return width > 0 && height > 0; }
};

struct DigitalMonitorDescriptor {
    int index = -1;
    std::string id;
    std::string friendly_name;
    DigitalRect desktop_rect{};
    DigitalRect work_rect{};
    bool is_primary = false;
    unsigned int dpi_x = 96;
    unsigned int dpi_y = 96;
    float scale_factor = 1.0f;
};

struct DigitalCaptureRequest {
    DigitalCaptureMode mode = DigitalCaptureMode::ActiveMonitor;
    int monitor_index = -1;
    bool include_layered_windows = true;
};

struct DigitalCaptureMetadata {
    DigitalCaptureStatus status = DigitalCaptureStatus::CaptureFailed;
    std::string error;
    std::string backend;
    std::string source_device_id;
    std::string source_platform;
    std::string source_transport;
    DigitalCaptureMode mode = DigitalCaptureMode::ActiveMonitor;

    int monitor_index = -1;
    std::string monitor_id;
    DigitalRect source_rect{};
    unsigned int dpi_x = 96;
    unsigned int dpi_y = 96;
    float scale_factor = 1.0f;

    std::string active_window_title;
    std::string active_process_name;
    DigitalRect active_window_rect{};

    std::uint64_t capture_steady_ns = 0;
    std::uint64_t capture_wall_ns = 0;
    double capture_duration_ms = 0.0;
    std::string color_space = "BGR8_SRGB";
};

struct DigitalCaptureResult {
    cv::Mat image;
    DigitalCaptureMetadata metadata{};

    bool Succeeded() const noexcept {
        return metadata.status == DigitalCaptureStatus::Ok && !image.empty();
    }
};

const char* ToString(DigitalCaptureMode mode) noexcept;
const char* ToString(DigitalCaptureStatus status) noexcept;

}}} // namespace GRIM::Perception::Digital
