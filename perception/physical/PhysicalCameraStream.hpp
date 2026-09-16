#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "PhysicalCameraFocusController.hpp"
#include "PhysicalCameraExposureController.hpp"

namespace GRIM { namespace Perception { namespace Physical {

enum class PhysicalCameraStreamState : uint8_t {
    Idle      = 0,
    Opening   = 1,
    Streaming = 2,
    Failed    = 3,
    Closed    = 4
};

struct PhysicalCapturedCameraFrame {
    cv::Mat   image;
    uint64_t  frame_counter    = 0;
    uint64_t  capture_steady_ns = 0;
    uint64_t  capture_wall_ns   = 0;
};

struct PhysicalCameraMotionExposureStatus {
    bool configured = false;
    bool last_set_accepted = false;
    double requested_exposure = 0.0;
    double requested_gain = 0.0;
    double negotiated_exposure = 0.0;
    double negotiated_gain = 0.0;
    double motion_priority = 0.0;
    std::string summary;
};

// One live IP camera connection, driven by a worker thread.
//
// API contract:
//   - OpenPhysicalCameraStream(url) returns immediately; opening happens on
//     the worker. Poll GetState() to see when it transitions to Streaming.
//   - PullLatestFrameInto() copies the most recent frame (if any) into the
//     caller's cv::Mat. Returns true if a NEW frame was copied since the
//     previous call (compared via internal frame counter).
//   - Any error sets state=Failed and stores a detailed reason retrievable
//     via GetLastErrorReason(). The class does NOT silently retry — the
//     owner (PhysicalEnvironmentLoop) decides whether to reopen.
class PhysicalCameraStream {
public:
    PhysicalCameraStream();
    ~PhysicalCameraStream();

    // Non-copyable, non-movable — owns a live thread.
    PhysicalCameraStream(const PhysicalCameraStream&)            = delete;
    PhysicalCameraStream& operator=(const PhysicalCameraStream&) = delete;

    // Throws if `url` is empty (Rule 20).
    void OpenPhysicalCameraStream(const std::string& url);

    // Synchronously stops the worker thread and releases the capture.
    void ClosePhysicalCameraStream();

    PhysicalCameraStreamState GetState() const;
    std::string               GetSourceUrl() const;
    std::string               GetLastErrorReason() const;
    uint64_t                  GetFrameCounter() const;
    double                    GetMeasuredFps() const;
    PhysicalCameraFocusStatus GetPhysicalCameraFocusStatusSnapshot() const;
    PhysicalCameraMotionExposureStatus
        GetPhysicalCameraMotionExposureStatusSnapshot() const;
    void RequestPhysicalCameraCalibrationFocusLock(bool locked);
    void RequestPhysicalCameraMotionExposure(
        const PhysicalCameraMotionExposureRequest& request);

    // Copies the most recent decoded frame into `out`. Returns true if the
    // frame is newer than what the caller saw last (tracked via
    // `last_seen_counter`). Returns false (and leaves `out` untouched) if
    // there is no frame yet OR if the frame counter has not advanced.
    bool PullLatestFrameInto(cv::Mat& out, uint64_t& last_seen_counter) const;

    // Copies the oldest retained frame newer than `last_seen_counter`.
    // The worker retains a bounded queue specifically so a stereo coordinator
    // can pair frames by capture timestamp instead of racing two latest slots.
    bool PullNextCapturedFrameInto(PhysicalCapturedCameraFrame& out,
                                   uint64_t& last_seen_counter) const;

private:
    void RunCaptureWorker(); // worker entry point

    std::string                                   source_url_;
    std::atomic<PhysicalCameraStreamState>        state_{PhysicalCameraStreamState::Idle};
    std::atomic<bool>                             stop_requested_{false};
    std::atomic<uint64_t>                         frame_counter_{0};
    std::atomic<double>                           measured_fps_{0.0};
    std::atomic<bool>                             calibration_focus_lock_requested_{false};
    PhysicalCameraFocusController                 focus_controller_;
    std::atomic<double>                           motion_exposure_priority_{0.0};
    std::atomic<double>                           motion_gain_demand_{0.0};
    std::atomic<uint64_t>                         motion_exposure_request_counter_{0};
    mutable std::mutex                            motion_exposure_mutex_;
    PhysicalCameraMotionExposureStatus            motion_exposure_status_{};

    mutable std::mutex                            frame_mutex_;
    cv::Mat                                       latest_frame_; // BGR8
    std::deque<PhysicalCapturedCameraFrame>       captured_frames_;
    mutable std::mutex                            error_mutex_;
    std::string                                   last_error_reason_;

    std::thread                                   worker_;
};

}}} // namespace GRIM::Perception::Physical
