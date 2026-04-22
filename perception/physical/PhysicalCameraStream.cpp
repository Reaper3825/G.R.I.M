#include "PhysicalCameraStream.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// ── Low-latency FFMPEG options ──
// OpenCV's FFMPEG backend buffers decoded frames in an internal FIFO. For
// RTSP/HTTP streams that defaults to ~3 seconds of glass-to-display latency
// because the consumer reads one frame per pull while the decoder fills the
// queue at the source FPS — the queue never drains. The drain pattern in
// the capture loop fixes the steady-state, but we ALSO want the decoder to
// stop pre-buffering in the first place.
//
// These options are read by OpenCV BEFORE cv::VideoCapture::open() runs, so
// we must set them prior to constructing the capture object. We respect any
// operator-provided value already in the environment.
//
//   fflags=nobuffer       — drop FFMPEG's internal buffering
//   flags=low_delay       — request low-delay decoding from the codec
//   rtsp_transport=tcp    — TCP avoids reorder-buffering vs UDP packet loss
//   max_delay=100000      — cap demuxer reorder buffer to 100ms (microseconds)
//   reorder_queue_size=0  — RTP-specific: no reorder buffer
//
// All separated by '|', key/value separated by ';' — OpenCV's parser format.
constexpr const char* kFfmpegLowLatencyOpts =
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
    "max_delay;100000|reorder_queue_size;0";

void EnsureFfmpegLowLatencyEnv() {
    if (const char* existing = std::getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS")) {
        // Respect operator override; only log so the difference is visible.
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  std::string("PhysicalCameraStream: using operator-provided "
                              "OPENCV_FFMPEG_CAPTURE_OPTIONS='") + existing + "'");
        return;
    }
#if defined(_WIN32)
    _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", kFfmpegLowLatencyOpts);
#else
    ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", kFfmpegLowLatencyOpts, /*overwrite=*/0);
#endif
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              std::string("PhysicalCameraStream: set OPENCV_FFMPEG_CAPTURE_OPTIONS='")
              + kFfmpegLowLatencyOpts + "'");
}

} // anonymous

PhysicalCameraStream::PhysicalCameraStream() = default;

PhysicalCameraStream::~PhysicalCameraStream() {
    ClosePhysicalCameraStream();
}

void PhysicalCameraStream::OpenPhysicalCameraStream(const std::string& url) {
    if (url.empty()) {
        throw std::runtime_error(
            "PhysicalCameraStream::OpenPhysicalCameraStream: url is empty — "
            "caller MUST provide a valid RTSP/HTTP URL");
    }
    // Hard refuse to leak a worker if Open is called twice.
    if (state_.load() == PhysicalCameraStreamState::Streaming
        || state_.load() == PhysicalCameraStreamState::Opening) {
        throw std::runtime_error(
            "PhysicalCameraStream::OpenPhysicalCameraStream: stream already "
            "Opening/Streaming — call ClosePhysicalCameraStream first");
    }

    // Make sure any previous worker is fully torn down.
    ClosePhysicalCameraStream();

    source_url_     = url;
    stop_requested_ = false;
    frame_counter_  = 0;
    measured_fps_   = 0.0;
    {
        std::lock_guard<std::mutex> lk(error_mutex_);
        last_error_reason_.clear();
    }
    state_  = PhysicalCameraStreamState::Opening;

    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "OpenPhysicalCameraStream: starting worker for url='" + url + "'");

    worker_ = std::thread(&PhysicalCameraStream::RunCaptureWorker, this);
}

void PhysicalCameraStream::ClosePhysicalCameraStream() {
    if (worker_.joinable()) {
        stop_requested_ = true;
        worker_.join();
    }
    state_ = PhysicalCameraStreamState::Closed;
}

PhysicalCameraStreamState PhysicalCameraStream::GetState() const {
    return state_.load();
}

std::string PhysicalCameraStream::GetSourceUrl() const {
    return source_url_;
}

std::string PhysicalCameraStream::GetLastErrorReason() const {
    std::lock_guard<std::mutex> lk(error_mutex_);
    return last_error_reason_;
}

uint64_t PhysicalCameraStream::GetFrameCounter() const {
    return frame_counter_.load();
}

double PhysicalCameraStream::GetMeasuredFps() const {
    return measured_fps_.load();
}

bool PhysicalCameraStream::PullLatestFrameInto(cv::Mat& out,
                                                uint64_t& last_seen_counter) const {
    const uint64_t current = frame_counter_.load();
    if (current == 0 || current == last_seen_counter) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        if (latest_frame_.empty()) return false;
        latest_frame_.copyTo(out);
    }
    last_seen_counter = current;
    return true;
}

void PhysicalCameraStream::RunCaptureWorker() {
    // Apply low-latency FFMPEG options BEFORE constructing the capture.
    EnsureFfmpegLowLatencyEnv();

    cv::VideoCapture cap;

    // Prefer FFMPEG backend for RTSP/HTTP URLs. If OpenCV was built without
    // FFMPEG, fall back to the default backend — but we surface that fact
    // in the error reason if open fails so the user knows immediately.
    //
    // Special case: url of the form "device:N" opens the Nth locally-attached
    // camera (USB/builtin) via the OS-default VideoCapture backend. This is
    // how physical cameras directly plugged into the host machine are used.
    bool opened = false;
    try {
        if (source_url_.rfind("device:", 0) == 0) {
            const std::string idx_str = source_url_.substr(7);
            if (idx_str.empty()) {
                throw std::runtime_error(
                    "device URL has empty index — expected 'device:<N>'");
            }
            const int device_index = std::stoi(idx_str);
            if (device_index < 0) {
                throw std::runtime_error(
                    "device index must be >= 0, got " + std::to_string(device_index));
            }
            // Explicit AVFoundation backend on macOS — CAP_ANY can pick a
            // slower fallback path on some OpenCV builds. On non-mac POSIX
            // (Linux V4L2) this falls through to CAP_ANY since
            // CAP_AVFOUNDATION is a no-op there.
#if defined(__APPLE__)
            opened = cap.open(device_index, cv::CAP_AVFOUNDATION);
            if (!opened) opened = cap.open(device_index, cv::CAP_ANY);
#else
            opened = cap.open(device_index, cv::CAP_ANY);
#endif
        } else {
            opened = cap.open(source_url_, cv::CAP_FFMPEG);
            if (!opened) {
                opened = cap.open(source_url_, cv::CAP_ANY);
            }
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(error_mutex_);
        last_error_reason_ = std::string("cv::VideoCapture::open threw: ") + e.what();
        state_ = PhysicalCameraStreamState::Failed;
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: open threw url='" + source_url_
                  + "' what=" + e.what());
        return;
    }

    if (!opened) {
        {
            std::lock_guard<std::mutex> lk(error_mutex_);
            last_error_reason_ =
                "cv::VideoCapture::open returned false for url='" + source_url_
                + "' (tried CAP_FFMPEG then CAP_ANY)";
        }
        state_ = PhysicalCameraStreamState::Failed;
        LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: open failed url='" + source_url_ + "'");
        return;
    }

    // Best-effort: ask the backend to keep its internal queue at a single
    // frame. AVFoundation honours this; FFMPEG/V4L2 may silently ignore it
    // (which is why the drain pattern below exists as the actual safety net).
    // We do not throw on failure — `set` returns false for unsupported props
    // on most backends, and that is not a real error condition.
    if (!cap.set(cv::CAP_PROP_BUFFERSIZE, 1)) {
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: backend rejected "
                  "CAP_PROP_BUFFERSIZE=1 — relying on drain pattern only");
    }

    // For local OS camera devices, ask the backend for 30 fps at 640x480.
    // AVFoundation on macOS picks an arbitrary default activeFormat that
    // often lands on 15 fps for the builtin FaceTime cam at 1080p/720p.
    // 640x480 is almost always available at 30+ fps on every builtin and
    // USB webcam, AND it cuts OpenCV's per-frame BGRA→BGR copy cost by 4x
    // on the worker thread — that copy is the real CPU bottleneck for the
    // OpenCV AVF backend on Apple Silicon, not the camera or memory bandwidth.
    // Network sources (RTSP/HTTP) ignore these properties — set() returns
    // false silently and we move on.
    if (source_url_.rfind("device:", 0) == 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS,          30);
        const double negotiated_fps = cap.get(cv::CAP_PROP_FPS);
        const double negotiated_w   = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double negotiated_h   = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: device negotiated "
                  + std::to_string(static_cast<int>(negotiated_w)) + "x"
                  + std::to_string(static_cast<int>(negotiated_h))
                  + " @ " + std::to_string(negotiated_fps) + " fps "
                  + "(if fps < 30 in good light, AVFoundation chose a "
                  + "lower-fps activeFormat; in low light, auto-exposure "
                  + "is lengthening shutter past 1/30s — brighten the room)");
    }

    state_ = PhysicalCameraStreamState::Streaming;
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: streaming url='" + source_url_ + "'");

    auto fps_window_start = std::chrono::steady_clock::now();
    uint64_t fps_window_frames = 0;

    // ── Drain pattern ──
    // Live RTSP/HTTP capture latency is a queue-depth problem, not a
    // throughput problem. The producer (camera/decoder) and the consumer
    // (this worker → frame bus → UI) both run at the source FPS, so the
    // FFMPEG internal queue NEVER drains naturally — whatever depth it
    // built up at startup persists forever (typically 60-90 frames =
    // 2-3 seconds of glass-to-display lag).
    //
    // Fix: each iteration we call grab() repeatedly to advance through any
    // already-decoded frames in the FFMPEG FIFO. grab() is cheap (no full
    // decode), so we can drain a backlog quickly. We stop draining when:
    //   1. grab() blocks for ~one inter-frame interval — that means the
    //      FIFO is empty and grab() is now waiting on the next live frame,
    //      i.e. we've caught up to "now"; OR
    //   2. we have drained kMaxDrainPerIter frames (defensive cap to avoid
    //      starving the rest of the loop on a misbehaving stream).
    // Then retrieve() decodes only that final, freshest frame.
    constexpr int  kMaxDrainPerIter        = 64;
    constexpr auto kFastGrabThresholdMicros = std::chrono::microseconds(8000); // 8ms
    cv::Mat scratch;
    while (!stop_requested_.load()) {
        // First grab: must succeed; this blocks for the next available frame.
        if (!cap.grab()) {
            std::lock_guard<std::mutex> lk(error_mutex_);
            last_error_reason_ =
                "cv::VideoCapture::grab returned false (stream ended or network drop)";
            state_ = PhysicalCameraStreamState::Failed;
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                      "PhysicalCameraStream worker: grab failed url='"
                      + source_url_ + "' after " + std::to_string(frame_counter_.load())
                      + " frames");
            cap.release();
            return;
        }

        // Drain any additional already-buffered frames. As long as grab()
        // returns "fast", there was a frame waiting in the FIFO. The first
        // grab() call that takes ~one inter-frame interval indicates an
        // empty FIFO — we abort it via the time check (we cannot truly
        // cancel mid-grab, so the cost of the final slow grab is bounded
        // by one frame interval and absorbed into next iteration's freshness).
        for (int drained = 0; drained < kMaxDrainPerIter; ++drained) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!cap.grab()) break;
            const auto dt = std::chrono::steady_clock::now() - t0;
            if (dt > kFastGrabThresholdMicros) {
                // That last grab() actually waited — we just consumed the
                // newest live frame. Fall through to retrieve it.
                break;
            }
        }

        if (!cap.retrieve(scratch) || scratch.empty()) {
            std::lock_guard<std::mutex> lk(error_mutex_);
            last_error_reason_ =
                "cv::VideoCapture::retrieve returned empty frame after grab";
            state_ = PhysicalCameraStreamState::Failed;
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG,
                      "PhysicalCameraStream worker: retrieve failed url='"
                      + source_url_ + "' after " + std::to_string(frame_counter_.load())
                      + " frames");
            cap.release();
            return;
        }

        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            scratch.copyTo(latest_frame_);
        }
        ++frame_counter_;
        ++fps_window_frames;

        auto now = std::chrono::steady_clock::now();
        auto window_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - fps_window_start).count();
        if (window_ms >= 1000) {
            measured_fps_.store(static_cast<double>(fps_window_frames)
                                * 1000.0 / static_cast<double>(window_ms));
            fps_window_start = now;
            fps_window_frames = 0;
        }
    }

    cap.release();
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: stop requested url='" + source_url_
              + "' total frames=" + std::to_string(frame_counter_.load()));
}

}}} // namespace
