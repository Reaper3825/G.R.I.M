#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// Driver-level focus request for a locally attached camera. A manual focus
// request always disables autofocus before setting the lens position.
struct PhysicalCameraFocusConfig {
    bool   configure_autofocus = false;
    bool   autofocus           = true;
    bool   configure_focus     = false;
    double manual_focus        = 0.0;
};

// Snapshot safe to read from the UI/main thread while capture is running.
// Property setters are best-effort because OpenCV camera backends and UVC
// drivers differ substantially in what they expose.
struct PhysicalCameraFocusStatus {
    bool        configuration_attempted = false;
    bool        autofocus_requested     = false;
    bool        autofocus_enabled       = false;
    bool        autofocus_set_accepted  = false;
    bool        manual_focus_requested  = false;
    bool        manual_focus_set_accepted = false;
    double      requested_focus         = 0.0;
    double      negotiated_autofocus    = 0.0;
    double      negotiated_focus        = 0.0;
    bool        calibration_focus_locked = false;
    bool        calibration_lock_accepted = false;
    double      focus_score             = 0.0; // variance of Laplacian on a small raw-frame view
    uint64_t    scored_frame_counter    = 0;
    std::string summary;
};

// Encapsulates optical focus control and focus telemetry. PhysicalCameraStream
// calls this class from its existing capture path; no second camera owner or
// side-channel is created.
class PhysicalCameraFocusController {
public:
    void Reset();

    // Apply focus properties after the camera has opened and its capture mode
    // has been negotiated. Unsupported properties are reported, not fatal.
    PhysicalCameraFocusStatus ApplyPhysicalCameraFocus(
        cv::VideoCapture& capture,
        const PhysicalCameraFocusConfig& config);

    // Freeze the current optical focus while calibration samples are being
    // collected, then restore the operator's configured focus mode.
    PhysicalCameraFocusStatus SetPhysicalCameraCalibrationFocusLock(
        cv::VideoCapture& capture,
        bool locked);

    // Samples every eighth frame to keep focus telemetry inexpensive.
    void ObserveRawFrameForFocus(const cv::Mat& raw_bgr, uint64_t frame_counter);

    PhysicalCameraFocusStatus GetPhysicalCameraFocusStatusSnapshot() const;

private:
    mutable std::mutex        mutex_;
    PhysicalCameraFocusStatus status_{};
    PhysicalCameraFocusConfig configured_focus_{};
    bool                      have_configuration_ = false;
    double                    autofocus_before_calibration_lock_ = 0.0;
};

}}} // namespace GRIM::Perception::Physical
