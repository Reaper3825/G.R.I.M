#include "PhysicalCameraStream.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "logger.hpp"

#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

struct LocalDeviceOpenRequest {
    int device_index = -1;
    int backend      = -1;
    int width        = 640;
    int height       = 480;
    int fps          = 60;
    std::string fourcc = "MJPG";
    double auto_exposure = std::numeric_limits<double>::quiet_NaN();
    double exposure      = std::numeric_limits<double>::quiet_NaN();
};

int ParseStrictNonNegativeInt(const std::string& text, const std::string& field_name) {
    if (text.empty()) {
        throw std::runtime_error(field_name + " is empty");
    }
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error(
            field_name + " must be a plain integer, got '" + text + "'");
    }
    if (value < 0) {
        throw std::runtime_error(
            field_name + " must be >= 0, got " + std::to_string(value));
    }
    return value;
}

int ParseStrictPositiveInt(const std::string& text, const std::string& field_name) {
    const int value = ParseStrictNonNegativeInt(text, field_name);
    if (value <= 0) {
        throw std::runtime_error(
            field_name + " must be > 0, got " + std::to_string(value));
    }
    return value;
}

double ParseStrictFiniteDouble(const std::string& text, const std::string& field_name) {
    if (text.empty()) {
        throw std::runtime_error(field_name + " is empty");
    }
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::runtime_error(
            field_name + " must be a plain number, got '" + text + "'");
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            field_name + " must be finite, got '" + text + "'");
    }
    return value;
}

std::string ToLowerAscii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool IsLocalDeviceUrl(const std::string& source_url) {
    return source_url.rfind("device:", 0) == 0;
}

void ApplyLocalDeviceQueryParam(LocalDeviceOpenRequest& req,
                                const std::string& key,
                                const std::string& value)
{
    if (key.empty()) {
        throw std::runtime_error("device URL query contains an empty key");
    }
    const std::string k = ToLowerAscii(key);
    if (k == "backend") {
        req.backend = ParseStrictNonNegativeInt(value, "device backend");
    } else if (k == "width") {
        req.width = ParseStrictPositiveInt(value, "capture width");
    } else if (k == "height") {
        req.height = ParseStrictPositiveInt(value, "capture height");
    } else if (k == "fps") {
        req.fps = ParseStrictPositiveInt(value, "capture fps");
    } else if (k == "fourcc") {
        if (value.size() != 4) {
            throw std::runtime_error(
                "capture fourcc must be exactly 4 ASCII characters, got '" + value + "'");
        }
        req.fourcc = value;
    } else if (k == "auto_exposure") {
        req.auto_exposure = ParseStrictFiniteDouble(value, "capture auto_exposure");
    } else if (k == "exposure") {
        req.exposure = ParseStrictFiniteDouble(value, "capture exposure");
    } else {
        throw std::runtime_error(
            "device URL query key '" + key + "' is unsupported; supported keys are "
            "backend,width,height,fps,fourcc,auto_exposure,exposure");
    }
}

LocalDeviceOpenRequest ParseLocalDeviceOpenRequest(const std::string& source_url) {
    if (source_url.rfind("device:", 0) != 0) {
        throw std::runtime_error(
            "ParseLocalDeviceOpenRequest: source_url does not start with device:");
    }
    const std::string body = source_url.substr(7);
    const size_t query_pos = body.find('?');
    LocalDeviceOpenRequest req;
    req.device_index = ParseStrictNonNegativeInt(
        query_pos == std::string::npos ? body : body.substr(0, query_pos),
        "device index");
    if (query_pos == std::string::npos) {
        return req;
    }
    const std::string query = body.substr(query_pos + 1);
    size_t start = 0;
    while (start < query.size()) {
        const size_t amp = query.find('&', start);
        const std::string item = query.substr(
            start,
            amp == std::string::npos ? std::string::npos : amp - start);
        const size_t eq = item.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(
                "device URL query item must be key=value, got '" + item + "'");
        }
        ApplyLocalDeviceQueryParam(req, item.substr(0, eq), item.substr(eq + 1));
        if (amp == std::string::npos) break;
        start = amp + 1;
    }
    return req;
}

int MakeFourcc(const std::string& fourcc) {
    if (fourcc.size() != 4) {
        throw std::runtime_error("MakeFourcc: fourcc must be exactly 4 characters");
    }
    return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

std::string FourccToString(int fourcc) {
    std::string text;
    text.push_back(static_cast<char>( fourcc        & 0xFF));
    text.push_back(static_cast<char>((fourcc >>  8) & 0xFF));
    text.push_back(static_cast<char>((fourcc >> 16) & 0xFF));
    text.push_back(static_cast<char>((fourcc >> 24) & 0xFF));
    return text;
}

void SetLocalCaptureProperty(cv::VideoCapture& cap,
                             int prop_id,
                             double value,
                             const std::string& label)
{
    const bool accepted = cap.set(prop_id, value);
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: request " + label + "="
              + std::to_string(value) + (accepted ? " accepted" : " rejected"));
}

std::vector<int> BuildLocalOpenParams(const LocalDeviceOpenRequest& req) {
    return {
        cv::CAP_PROP_FOURCC, MakeFourcc(req.fourcc),
        cv::CAP_PROP_FRAME_WIDTH, req.width,
        cv::CAP_PROP_FRAME_HEIGHT, req.height,
        cv::CAP_PROP_FPS, req.fps,
        cv::CAP_PROP_BUFFERSIZE, 1
    };
}

bool OpenLocalDeviceWithRequestedMode(cv::VideoCapture& cap,
                                      const LocalDeviceOpenRequest& req,
                                      int backend)
{
    const std::vector<int> params = BuildLocalOpenParams(req);
    try {
        if (cap.open(req.device_index, backend, params)) {
            LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                      "PhysicalCameraStream worker: local open accepted mode params "
                      "device=" + std::to_string(req.device_index)
                      + " backend=" + std::to_string(backend)
                      + " " + std::to_string(req.width) + "x" + std::to_string(req.height)
                      + " @ " + std::to_string(req.fps) + " fps fourcc=" + req.fourcc);
            return true;
        }
    } catch (const std::exception& e) {
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: local open with mode params threw backend="
                  + std::to_string(backend) + " what=" + e.what()
                  + " — trying explicit post-open property negotiation next");
    }
    cap.release();
    if (!cap.open(req.device_index, backend)) {
        return false;
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: local open used post-open property negotiation "
              "device=" + std::to_string(req.device_index)
              + " backend=" + std::to_string(backend));
    return true;
}

std::vector<int> LocalDeviceBackendsToTry() {
    return {
#if defined(__APPLE__)
        cv::CAP_AVFOUNDATION,
#elif defined(_WIN32)
        cv::CAP_DSHOW,
        cv::CAP_MSMF,
#else
        cv::CAP_V4L2,
#endif
        cv::CAP_ANY
    };
}

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
    const bool is_local_device = IsLocalDeviceUrl(source_url_);

    // Prefer FFMPEG backend for RTSP/HTTP URLs. If OpenCV was built without
    // FFMPEG, fall back to the default backend — but we surface that fact
    // in the error reason if open fails so the user knows immediately.
    //
    // Special case: url of the form "device:N" opens the Nth locally-attached
    // camera (USB/builtin), while "device:N?backend=M" pins the same OpenCV
    // backend that enumeration verified. Backend pinning matters on Windows:
    // DirectShow and MediaFoundation can expose virtual + USB cameras in
    // different orders, so a source row must open the same backend it probed.
    bool opened = false;
    try {
        if (is_local_device) {
            const LocalDeviceOpenRequest req = ParseLocalDeviceOpenRequest(source_url_);
            if (req.backend >= 0) {
                opened = OpenLocalDeviceWithRequestedMode(cap, req, req.backend);
            } else {
                for (int backend : LocalDeviceBackendsToTry()) {
                    opened = OpenLocalDeviceWithRequestedMode(cap, req, backend);
                    if (opened) break;
                    cap.release();
                }
            }
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

    // For local OS camera devices, request a high-FPS, low-resolution capture
    // mode. The previous hard-coded 30 fps request capped every capable USB
    // camera at ~29.97 fps before perception/inference ever got a vote.
    // MJPG is requested before geometry/FPS because many Windows UVC devices
    // only expose 60 fps at 640x480 when using compressed transport; YUY2 is
    // commonly limited to 30 fps by USB bandwidth.
    //
    // Override per source with:
    //   device:N?backend=M&width=640&height=480&fps=120&fourcc=MJPG
    // Backend refusal is logged along with negotiated values; hardware/driver
    // capabilities still decide the final FPS.
    if (is_local_device) {
        const LocalDeviceOpenRequest req = ParseLocalDeviceOpenRequest(source_url_);
        SetLocalCaptureProperty(cap,
                                cv::CAP_PROP_FOURCC,
                                static_cast<double>(MakeFourcc(req.fourcc)),
                                "fourcc(" + req.fourcc + ")");
        SetLocalCaptureProperty(cap, cv::CAP_PROP_FRAME_WIDTH,
                                static_cast<double>(req.width), "width");
        SetLocalCaptureProperty(cap, cv::CAP_PROP_FRAME_HEIGHT,
                                static_cast<double>(req.height), "height");
        SetLocalCaptureProperty(cap, cv::CAP_PROP_FPS,
                                static_cast<double>(req.fps), "fps");
        if (std::isfinite(req.auto_exposure)) {
            SetLocalCaptureProperty(cap, cv::CAP_PROP_AUTO_EXPOSURE,
                                    req.auto_exposure, "auto_exposure");
        }
        if (std::isfinite(req.exposure)) {
            SetLocalCaptureProperty(cap, cv::CAP_PROP_EXPOSURE,
                                    req.exposure, "exposure");
        }
        const double negotiated_fps = cap.get(cv::CAP_PROP_FPS);
        const double negotiated_w   = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double negotiated_h   = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        const int negotiated_fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        const std::string negotiated_fourcc_text = FourccToString(negotiated_fourcc);
        const double negotiated_auto_exposure = cap.get(cv::CAP_PROP_AUTO_EXPOSURE);
        const double negotiated_exposure      = cap.get(cv::CAP_PROP_EXPOSURE);
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "PhysicalCameraStream worker: device requested "
                  + std::to_string(req.width) + "x" + std::to_string(req.height)
                  + " @ " + std::to_string(req.fps) + " fps fourcc=" + req.fourcc
                  + "; negotiated "
                  + std::to_string(static_cast<int>(negotiated_w)) + "x"
                  + std::to_string(static_cast<int>(negotiated_h))
                  + " @ " + std::to_string(negotiated_fps) + " fps fourcc="
                  + negotiated_fourcc_text
                  + " auto_exposure=" + std::to_string(negotiated_auto_exposure)
                  + " exposure=" + std::to_string(negotiated_exposure)
                  + " (if measured FPS remains near 30, the selected camera/backend "
                  + "may be throttling via actual pixel format, exposure, or driver timing)");
        if (negotiated_fourcc_text != req.fourcc) {
            LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                      "PhysicalCameraStream worker: requested fourcc=" + req.fourcc
                      + " but backend is delivering fourcc=" + negotiated_fourcc_text
                      + " — try another backend or explicit device URL; YUY2 often reports "
                      + "high FPS while the driver still clocks frames near 30");
        }
    }

    state_ = PhysicalCameraStreamState::Streaming;
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: streaming url='" + source_url_ + "'");

    auto fps_window_start = std::chrono::steady_clock::now();
    uint64_t fps_window_frames = 0;
    double fps_window_grab_ms = 0.0;
    double fps_window_retrieve_ms = 0.0;
    double fps_window_store_ms = 0.0;

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
    //
    // IMPORTANT: local `device:N` cameras are NOT drained this way. On UVC /
    // AVFoundation devices, the extra grab() blocks for the next live frame;
    // retrieving only after that second grab drops every other frame and turns
    // a real 60 FPS capture mode into ~29-30 FPS. Local devices use one
    // grab()+retrieve() per worker iteration; network streams keep the drain.
    constexpr int  kMaxDrainPerIter        = 64;
    constexpr auto kFastGrabThresholdMicros = std::chrono::microseconds(8000); // 8ms
    cv::Mat scratch;
    while (!stop_requested_.load()) {
        // First grab: must succeed; this blocks for the next available frame.
        const auto grab_start = std::chrono::steady_clock::now();
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
        fps_window_grab_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - grab_start).count();

        if (!is_local_device) {
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
        }

        const auto retrieve_start = std::chrono::steady_clock::now();
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
        fps_window_retrieve_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - retrieve_start).count();

        const auto store_start = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            scratch.copyTo(latest_frame_);
        }
        fps_window_store_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - store_start).count();
        ++frame_counter_;
        ++fps_window_frames;

        auto now = std::chrono::steady_clock::now();
        auto window_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - fps_window_start).count();
        if (window_ms >= 1000) {
            const double actual_fps = static_cast<double>(fps_window_frames)
                                      * 1000.0 / static_cast<double>(window_ms);
            measured_fps_.store(actual_fps);
            if (is_local_device && fps_window_frames > 0) {
                const double avg_grab_ms = fps_window_grab_ms
                                           / static_cast<double>(fps_window_frames);
                const double avg_retrieve_ms = fps_window_retrieve_ms
                                               / static_cast<double>(fps_window_frames);
                const double avg_store_ms = fps_window_store_ms
                                            / static_cast<double>(fps_window_frames);
                const double current_negotiated_fps = cap.get(cv::CAP_PROP_FPS);
                if (current_negotiated_fps >= 50.0 && actual_fps < current_negotiated_fps * 0.75) {
                    const int current_fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
                    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                              "PhysicalCameraStream worker: local FPS mismatch "
                              "negotiated=" + std::to_string(current_negotiated_fps)
                              + " measured=" + std::to_string(actual_fps)
                              + " avg_grab_ms=" + std::to_string(avg_grab_ms)
                              + " avg_retrieve_ms=" + std::to_string(avg_retrieve_ms)
                              + " avg_store_ms=" + std::to_string(avg_store_ms)
                              + " fourcc=" + FourccToString(current_fourcc)
                              + " auto_exposure=" + std::to_string(cap.get(cv::CAP_PROP_AUTO_EXPOSURE))
                              + " exposure=" + std::to_string(cap.get(cv::CAP_PROP_EXPOSURE))
                              + " — if avg_grab_ms is ~33ms, the backend/driver/camera "
                              + "is delivering 30Hz despite reporting a 60Hz mode");
                }
            }
            fps_window_start = now;
            fps_window_frames = 0;
            fps_window_grab_ms = 0.0;
            fps_window_retrieve_ms = 0.0;
            fps_window_store_ms = 0.0;
        }
    }

    cap.release();
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "PhysicalCameraStream worker: stop requested url='" + source_url_
              + "' total frames=" + std::to_string(frame_counter_.load()));
}

}}} // namespace
