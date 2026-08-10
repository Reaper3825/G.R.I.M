#include "PhysicalEnvironmentLoop.hpp"

#include "PhysicalCameraCalibrator.hpp"
#include "PhysicalCameraDirectory.hpp"
#include "PhysicalCameraStream.hpp"
#include "PhysicalFrameConditioner.hpp"
#include "PhysicalEnvironmentLogTag.hpp"
#include "PhysicalFrameBus.hpp"
#include "PhysicalStereoCapture.hpp"
#include "PhysicalStereoFrameBus.hpp"
#include "logger.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// All process-wide state for the physical environment subsystem lives here.
// Encapsulated in an anonymous namespace so no other translation unit can
// touch it directly — Rule 2 ("internally managed").
struct PhysicalEnvironmentState {
    std::mutex                              mutex;          // guards everything below
    bool                                    initialized      = false;
    bool                                    shutting_down    = false;
    const ::GRIM::DeviceCommServer*         device_server    = nullptr;
    std::vector<PhysicalCameraSource>       directory;
    std::unique_ptr<PhysicalCameraStream>   active_stream;
    std::unique_ptr<PhysicalStereoCapture>  stereo_capture;
    PhysicalStereoCaptureConfig             stereo_config{};
    std::string                             active_label;
    std::string                             last_error_reason;
    cv::Mat                                 pull_scratch;   // worker → bus copy buffer
    cv::Mat                                 model_scratch;  // conditioned signal for model path
    PhysicalFrameConditioner                conditioner;
    uint64_t                                last_pulled_counter = 0;
};

PhysicalEnvironmentState& GetState() {
    static PhysicalEnvironmentState s;
    return s;
}

void LazyInitLocked(PhysicalEnvironmentState& s) {
    if (s.initialized) return;
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "TickPhysicalEnvironment: first call — running lazy init");
    if (s.directory.empty()) {
        try {
            s.directory = RefreshPhysicalCameraDirectory(s.device_server);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("lazy directory refresh failed: ") + e.what();
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG, s.last_error_reason);
            // Initialization continues — the UI can re-request a refresh.
        }
    } else {
        LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
                  "TickPhysicalEnvironment: lazy init reusing preloaded camera directory ("
                  + std::to_string(s.directory.size()) + " entries)");
    }
    s.initialized = true;
}

void DrainActiveStreamLocked(PhysicalEnvironmentState& s) {
    if (!s.active_stream) return;
    const auto state = s.active_stream->GetState();

    if (state == PhysicalCameraStreamState::Failed) {
        const std::string reason         = s.active_stream->GetLastErrorReason();
        const std::string prefixed       = "active stream failed: " + reason;
        // Compare against the SAME prefixed string we'd store — otherwise the
        // prefix makes the dedup check always false and we spam the log.
        if (s.last_error_reason != prefixed) {
            s.last_error_reason = prefixed;
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG, s.last_error_reason);
        }
        return;
    }
    if (state != PhysicalCameraStreamState::Streaming) {
        return; // nothing to drain yet
    }

    if (s.active_stream->PullLatestFrameInto(s.pull_scratch, s.last_pulled_counter)) {
        try {
            const auto t_capture_steady = std::chrono::steady_clock::now();
            const auto t_capture_wall   = std::chrono::system_clock::now();

            const auto cond_result = s.conditioner.ProcessRawFrameToModelSignal(
                s.pull_scratch,
                s.last_pulled_counter,
                s.model_scratch);

            if (!cond_result.accepted) {
                // Quality gate dropped this frame. Do NOT publish — that's the
                // whole point: keep garbage out of the model's temporal context.
                s.last_error_reason = "frame dropped by quality gate: "
                                      + cond_result.drop_reason;
                // Debug-only: the conditioner already logged the reason.
                return;
            }

            PhysicalFrameMetadata md;
            md.capture_steady_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_capture_steady.time_since_epoch()).count());
            md.publish_steady_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            md.capture_wall_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_capture_wall.time_since_epoch()).count());
            md.raw_width             = cond_result.raw_width;
            md.raw_height            = cond_result.raw_height;
            md.model_width           = cond_result.model_width;
            md.model_height          = cond_result.model_height;
            md.raw_to_model          = cond_result.raw_to_model;
            md.color_space_label     = cond_result.color_space_label;
            md.pipeline_summary      = cond_result.pipeline_summary;
            md.applied_exposure_gain = s.conditioner
                .GetPhysicalSignalConditioningStatusSnapshot()
                .last_applied_exposure_gain;
            md.scene_stability       = cond_result.scene_stability;
            md.conditioning_total_ms           = cond_result.total_ms;
            md.conditioning_quality_gate_ms    = cond_result.quality_gate_ms;
            md.conditioning_stabilization_ms   = cond_result.stabilization_ms;
            md.conditioning_denoise_ms         = cond_result.denoise_ms;
            md.conditioning_exposure_ms        = cond_result.exposure_ms;
            md.conditioning_deblur_ms          = cond_result.deblur_ms;
            md.conditioning_resize_ms          = cond_result.resize_ms;
            md.conditioning_color_convert_ms   = cond_result.color_convert_ms;
            md.conditioning_scene_stability_ms = cond_result.scene_stability_ms;

            PhysicalFrameBus::Instance().PublishPhysicalFrameToBus(
                s.pull_scratch,
                s.model_scratch,
                s.last_pulled_counter,
                s.active_stream->GetSourceUrl(),
                s.active_label,
                md);
        } catch (const std::exception& e) {
            s.last_error_reason = std::string("publish to FrameBus failed: ") + e.what();
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG, s.last_error_reason);
        }
    }
}

void DrainStereoCaptureLocked(PhysicalEnvironmentState& s) {
    if (!s.stereo_capture) return;
    PhysicalStereoFramePair pair;
    if (s.stereo_capture->PullLatestSynchronizedPairInto(pair)) {
        PhysicalStereoFrameBus::Instance().PublishPhysicalStereoFramePairToBus(
            pair, s.stereo_config);
    }
    const auto status = s.stereo_capture->GetPhysicalStereoCaptureStatus();
    if (!status.last_error_reason.empty()) {
        const std::string prefixed = "stereo capture failed: " + status.last_error_reason;
        if (s.last_error_reason != prefixed) {
            s.last_error_reason = prefixed;
            LOG_ERROR(PHYSICAL_ENV_LOG_TAG, prefixed);
        }
    }
}

} // anonymous namespace

void RegisterPhysicalEnvironmentDeviceServer(const ::GRIM::DeviceCommServer* server) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.device_server = server;
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              std::string("RegisterPhysicalEnvironmentDeviceServer: server=")
              + (server ? "set" : "null"));
}

void TickPhysicalEnvironment() {
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        if (s.shutting_down) return;
        LazyInitLocked(s);
        DrainActiveStreamLocked(s);
        DrainStereoCaptureLocked(s);
    }
    // Calibrator runs OUTSIDE the env-loop mutex: it has its own mutex and
    // it consumes from the FrameBus (which is independently locked). This
    // keeps the single mainloop integration point (TickPhysicalEnvironment)
    // intact while the calibrator owns its lifecycle.
    TickPhysicalCameraCalibration();
}

void ShutdownPhysicalEnvironment() {
    auto& s = GetState();

    // Pull the unique_ptr out under lock then destroy it OUTSIDE the lock so
    // the worker thread (which may briefly want the lock for nothing — it
    // doesn't, but defensive) can join cleanly.
    std::unique_ptr<PhysicalCameraStream> to_destroy;
    std::unique_ptr<PhysicalStereoCapture> stereo_to_destroy;
    {
        std::lock_guard<std::mutex> lk(s.mutex);
        s.shutting_down = true;
        to_destroy      = std::move(s.active_stream);
        stereo_to_destroy = std::move(s.stereo_capture);
    }
    to_destroy.reset(); // joins the worker
    stereo_to_destroy.reset();

    {
        std::lock_guard<std::mutex> lk(s.mutex);
        PhysicalFrameBus::Instance().ResetPhysicalFrameBus();
        PhysicalStereoFrameBus::Instance().ResetPhysicalStereoFrameBus();
        s.conditioner.ResetPhysicalSignalConditioningTemporalState();
        s.directory.clear();
        s.active_label.clear();
        s.initialized = false;
    }
    // Drop calibrator state too — its lazy-init will re-load from disk on
    // the next TickPhysicalEnvironment() call.
    ResetPhysicalCalibrationState();
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "ShutdownPhysicalEnvironment: complete");
}

bool IsPhysicalEnvironmentRunning() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.initialized && !s.shutting_down;
}

// ── Read-only state for UI ──────────────────────────────────────────────────

std::vector<PhysicalCameraSource> GetPhysicalCameraDirectorySnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.directory;
}

std::string GetActivePhysicalCameraUrl() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_stream ? s.active_stream->GetSourceUrl() : std::string{};
}

std::string GetActivePhysicalCameraLabel() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_label;
}

std::string GetLastEnvironmentError() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.last_error_reason;
}

bool IsPhysicalCameraStreamActive() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_stream
           && s.active_stream->GetState() == PhysicalCameraStreamState::Streaming;
}

bool IsPhysicalCameraStreamFailed() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_stream
           && s.active_stream->GetState() == PhysicalCameraStreamState::Failed;
}

double GetActiveStreamFps() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_stream ? s.active_stream->GetMeasuredFps() : 0.0;
}

uint64_t GetActiveStreamFrameCounter() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.active_stream ? s.active_stream->GetFrameCounter() : 0;
}

PhysicalStereoCaptureStatus GetPhysicalStereoCaptureStatusSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    if (!s.stereo_capture) return PhysicalStereoCaptureStatus{};
    return s.stereo_capture->GetPhysicalStereoCaptureStatus();
}

bool IsPhysicalStereoCaptureActive() {
    const auto status = GetPhysicalStereoCaptureStatusSnapshot();
    return status.left_state == PhysicalCameraStreamState::Streaming
        && status.right_state == PhysicalCameraStreamState::Streaming;
}

PhysicalSignalConditioningConfig GetPhysicalSignalConditioningConfigSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.conditioner.GetPhysicalSignalConditioningConfigSnapshot();
}

PhysicalSignalConditioningStatus GetPhysicalSignalConditioningStatusSnapshot() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    return s.conditioner.GetPhysicalSignalConditioningStatusSnapshot();
}

// ── UI-facing requests ──────────────────────────────────────────────────────

void RequestPhysicalCameraDirectoryRefresh() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.directory = RefreshPhysicalCameraDirectory(s.device_server); // throws on OS error
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "RequestPhysicalCameraDirectoryRefresh: " + std::to_string(s.directory.size())
              + " entries");
}

void RequestOpenPhysicalCameraSource(const std::string& url,
                                     const std::string& display_label) {
    if (url.empty()) {
        throw std::runtime_error(
            "RequestOpenPhysicalCameraSource: url is empty — caller MUST provide a URL");
    }

    std::unique_ptr<PhysicalCameraStream> previous;
    std::unique_ptr<PhysicalStereoCapture> previous_stereo;
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        previous              = std::move(s.active_stream);
        previous_stereo       = std::move(s.stereo_capture);
        s.active_label        = display_label;
        s.last_error_reason.clear();
        s.last_pulled_counter = 0;
    }
    previous.reset(); // join old worker outside lock
    previous_stereo.reset();
    PhysicalFrameBus::Instance().ResetPhysicalFrameBus();
    PhysicalStereoFrameBus::Instance().ResetPhysicalStereoFrameBus();
    ResetPhysicalCalibrationState();
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.conditioner.ResetPhysicalSignalConditioningTemporalState();
    }

    auto fresh = std::make_unique<PhysicalCameraStream>();
    fresh->OpenPhysicalCameraStream(url); // throws if url empty (already checked)

    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.active_stream = std::move(fresh);
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG,
              "RequestOpenPhysicalCameraSource: opened url='" + url + "' label='"
              + display_label + "'");
}

void RequestClosePhysicalCameraSource() {
    std::unique_ptr<PhysicalCameraStream> previous;
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        previous       = std::move(s.active_stream);
        s.active_label.clear();
        s.last_pulled_counter = 0;
    }
    previous.reset();
    PhysicalFrameBus::Instance().ResetPhysicalFrameBus();
    ResetPhysicalCalibrationState();
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.conditioner.ResetPhysicalSignalConditioningTemporalState();
    }
    LOG_DEBUG(PHYSICAL_ENV_LOG_TAG, "RequestClosePhysicalCameraSource: closed");
}

void RequestOpenPhysicalStereoCameraPair(const PhysicalStereoCaptureConfig& config) {
    std::unique_ptr<PhysicalCameraStream> previous_mono;
    std::unique_ptr<PhysicalStereoCapture> previous_stereo;
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        previous_mono   = std::move(s.active_stream);
        previous_stereo = std::move(s.stereo_capture);
        s.active_label.clear();
        s.last_pulled_counter = 0;
        s.last_error_reason.clear();
    }
    previous_mono.reset();
    previous_stereo.reset();
    PhysicalFrameBus::Instance().ResetPhysicalFrameBus();
    PhysicalStereoFrameBus::Instance().ResetPhysicalStereoFrameBus();
    ResetPhysicalCalibrationState();

    auto fresh = std::make_unique<PhysicalStereoCapture>();
    fresh->OpenPhysicalStereoCapture(config);
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        s.stereo_config  = config;
        s.stereo_capture = std::move(fresh);
        s.conditioner.ResetPhysicalSignalConditioningTemporalState();
    }
}

void RequestClosePhysicalStereoCameraPair() {
    std::unique_ptr<PhysicalStereoCapture> previous;
    {
        auto& s = GetState();
        std::lock_guard<std::mutex> lk(s.mutex);
        previous = std::move(s.stereo_capture);
    }
    previous.reset();
    PhysicalStereoFrameBus::Instance().ResetPhysicalStereoFrameBus();
    ResetPhysicalCalibrationState();
}

void RequestConfigurePhysicalSignalConditioning(const PhysicalSignalConditioningConfig& config) {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.conditioner.ConfigurePhysicalSignalConditioning(config);
}

void RequestResetPhysicalSignalConditioningDefaults() {
    auto& s = GetState();
    std::lock_guard<std::mutex> lk(s.mutex);
    s.conditioner.ResetPhysicalSignalConditioningToDefaults();
}

}}} // namespace
