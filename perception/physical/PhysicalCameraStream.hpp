#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

enum class PhysicalCameraStreamState : uint8_t {
    Idle      = 0,
    Opening   = 1,
    Streaming = 2,
    Failed    = 3,
    Closed    = 4
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

    // Copies the most recent decoded frame into `out`. Returns true if the
    // frame is newer than what the caller saw last (tracked via
    // `last_seen_counter`). Returns false (and leaves `out` untouched) if
    // there is no frame yet OR if the frame counter has not advanced.
    bool PullLatestFrameInto(cv::Mat& out, uint64_t& last_seen_counter) const;

private:
    void RunCaptureWorker(); // worker entry point

    std::string                                   source_url_;
    std::atomic<PhysicalCameraStreamState>        state_{PhysicalCameraStreamState::Idle};
    std::atomic<bool>                             stop_requested_{false};
    std::atomic<uint64_t>                         frame_counter_{0};
    std::atomic<double>                           measured_fps_{0.0};

    mutable std::mutex                            frame_mutex_;
    cv::Mat                                       latest_frame_; // BGR8
    mutable std::mutex                            error_mutex_;
    std::string                                   last_error_reason_;

    std::thread                                   worker_;
};

}}} // namespace GRIM::Perception::Physical
