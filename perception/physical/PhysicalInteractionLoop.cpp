#include "PhysicalInteractionLoop.hpp"

#include "PhysicalFrameBus.hpp"
#include "PhysicalHandGestureBus.hpp"
#include "logger.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

constexpr const char* kLogTag = "PHYSICAL_INTERACTION";

uint64_t SteadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string DefaultGestureModelPath() {
#if defined(GRIM_ROOT_DIR)
    return (std::filesystem::path(GRIM_ROOT_DIR) / "resources" / "models" /
            "vision" / "mediapipe" / "gesture_recognizer.task").string();
#else
    return (std::filesystem::path("resources") / "models" / "vision" /
            "mediapipe" / "gesture_recognizer.task").string();
#endif
}

struct PhysicalInteractionState {
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false;
    bool stop_requested = false;
    bool backend_ready = false;
    uint64_t config_generation = 1;
    uint64_t last_seen_frame_counter = 0;
    uint64_t last_submitted_capture_ns = 0;

    PhysicalHandGestureConfig config{};
    PhysicalHandGestureSnapshot snapshot{};
    std::unique_ptr<IPhysicalHandGestureBackend> backend;
    std::optional<PhysicalFrameBus::FrameView> pending_frame;
    std::thread worker;

    PhysicalInteractionState() {
        config.model_path = DefaultGestureModelPath();
        snapshot.model_path = config.model_path;
        snapshot.status_detail = "Waiting for interaction worker initialization";
    }
};

PhysicalInteractionState& State() {
    static PhysicalInteractionState state;
    return state;
}

void PublishSnapshotLocked(PhysicalInteractionState& state) {
    state.snapshot.enabled = state.config.enabled;
    state.snapshot.model_path = state.config.model_path;
    state.snapshot.offline_only = true;
    PhysicalHandGestureBus::Instance().PublishPhysicalHandGestureSnapshot(
        state.snapshot);
}

void SetBackendIdentityLocked(PhysicalInteractionState& state) {
    if (!state.backend) return;
    state.snapshot.backend_name = state.backend->BackendName();
    state.snapshot.backend_version = state.backend->BackendVersion();
    state.snapshot.telemetry_disabled = state.backend->TelemetryDisabled();
}

void ApplyBackendConfiguration(PhysicalInteractionState& state,
                               uint64_t generation,
                               const PhysicalHandGestureConfig& config)
{
    state.backend->Shutdown();

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.backend_ready = false;
        SetBackendIdentityLocked(state);
        state.snapshot.hands.clear();
        state.snapshot.source_frame_counter = 0;
        state.snapshot.last_error.clear();
        state.snapshot.worker_busy = true;
        state.snapshot.backend_state = config.enabled
            ? PhysicalHandGestureBackendState::Initializing
            : PhysicalHandGestureBackendState::Disabled;
        state.snapshot.status_detail = config.enabled
            ? "Initializing local MediaPipe gesture backend"
            : "Hand gesture recognition is disabled";
        PublishSnapshotLocked(state);
    }

    if (!config.enabled) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.worker_busy = false;
        PublishSnapshotLocked(state);
        return;
    }

    if (!state.backend->IsCompiled()) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.backend_state =
            PhysicalHandGestureBackendState::BackendUnavailable;
        state.snapshot.status_detail =
            "Optional MediaPipe backend is not present in this build";
        state.snapshot.last_error = state.snapshot.status_detail;
        state.snapshot.worker_busy = false;
        PublishSnapshotLocked(state);
        return;
    }

    if (!state.backend->TelemetryDisabled()) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.backend_state = PhysicalHandGestureBackendState::Failed;
        state.snapshot.status_detail =
            "Refusing MediaPipe backend: offline telemetry guarantee is absent";
        state.snapshot.last_error = state.snapshot.status_detail;
        state.snapshot.worker_busy = false;
        PublishSnapshotLocked(state);
        return;
    }

    std::error_code model_path_error;
    const bool local_model_exists = !config.model_path.empty() &&
        config.model_path.find("://") == std::string::npos &&
        std::filesystem::is_regular_file(config.model_path, model_path_error);
    if (!local_model_exists) {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.backend_state = PhysicalHandGestureBackendState::ModelMissing;
        state.snapshot.status_detail =
            "Place a local gesture_recognizer.task model at the configured path";
        state.snapshot.last_error = "Local model unavailable: " + config.model_path;
        state.snapshot.worker_busy = false;
        PublishSnapshotLocked(state);
        return;
    }

    std::string error;
    const bool initialized = state.backend->Initialize(config, error);
    std::lock_guard<std::mutex> lock(state.mutex);
    // If configuration changed while initialization was running, leave this
    // instance unused; the worker will immediately apply the newer generation.
    if (generation != state.config_generation) {
        state.backend->Shutdown();
        state.snapshot.worker_busy = false;
        return;
    }
    state.backend_ready = initialized;
    state.snapshot.backend_state = initialized
        ? PhysicalHandGestureBackendState::Ready
        : PhysicalHandGestureBackendState::Failed;
    state.snapshot.status_detail = initialized
        ? "Ready; inference runs locally on the bounded interaction worker"
        : "MediaPipe backend initialization failed";
    state.snapshot.last_error = initialized ? std::string{} : error;
    state.snapshot.worker_busy = false;
    PublishSnapshotLocked(state);
}

void WorkerMain() {
    auto& state = State();
    uint64_t applied_generation = 0;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.worker_running = true;
        SetBackendIdentityLocked(state);
        PublishSnapshotLocked(state);
    }

    for (;;) {
        PhysicalHandGestureConfig config;
        PhysicalFrameBus::FrameView frame;
        uint64_t generation = 0;
        bool apply_config = false;
        bool process_frame = false;

        {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.cv.wait(lock, [&] {
                return state.stop_requested ||
                       applied_generation != state.config_generation ||
                       (state.backend_ready && state.pending_frame.has_value());
            });
            if (state.stop_requested) break;

            generation = state.config_generation;
            if (applied_generation != generation) {
                config = state.config;
                applied_generation = generation;
                apply_config = true;
            } else if (state.backend_ready && state.pending_frame) {
                frame = std::move(*state.pending_frame);
                state.pending_frame.reset();
                state.snapshot.worker_busy = true;
                PublishSnapshotLocked(state);
                process_frame = true;
            }
        }

        if (apply_config) {
            try {
                ApplyBackendConfiguration(state, generation, config);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.backend_ready = false;
                state.snapshot.backend_state =
                    PhysicalHandGestureBackendState::Failed;
                state.snapshot.worker_busy = false;
                state.snapshot.status_detail =
                    "Gesture backend configuration threw an exception";
                state.snapshot.last_error = e.what();
                PublishSnapshotLocked(state);
                LOG_ERROR(kLogTag, state.snapshot.last_error);
            }
            continue;
        }
        if (!process_frame) continue;

        cv::Mat rgb;
        std::vector<PhysicalHandObservation> observations;
        double inference_ms = 0.0;
        std::string error;
        bool ok = false;

        try {
            if (frame.raw_image.empty()) {
                error = "PhysicalFrameBus provided an empty raw image";
            } else {
                cv::cvtColor(frame.raw_image, rgb, cv::COLOR_BGR2RGB);
                if (!rgb.isContinuous()) rgb = rgb.clone();
                PhysicalHandGestureFrame input;
                input.rgb_data = rgb.ptr<uint8_t>(0);
                input.width = rgb.cols;
                input.height = rgb.rows;
                input.byte_count = static_cast<int>(rgb.total() * rgb.elemSize());
                input.source_frame_counter = frame.frame_counter;
                input.source_capture_steady_ns = frame.metadata.capture_steady_ns;
                ok = state.backend->Process(
                    input, observations, inference_ms, error);
            }
        } catch (const std::exception& e) {
            error = std::string("Gesture worker exception: ") + e.what();
            ok = false;
        }

        std::lock_guard<std::mutex> lock(state.mutex);
        state.snapshot.worker_busy = false;
        if (generation != state.config_generation) {
            state.snapshot.status_detail =
                "Discarded result produced under an older configuration";
            PublishSnapshotLocked(state);
            continue;
        }
        state.snapshot.last_inference_ms = inference_ms;
        if (ok) {
            ++state.snapshot.frames_processed;
            const double n = static_cast<double>(state.snapshot.frames_processed);
            state.snapshot.mean_inference_ms +=
                (inference_ms - state.snapshot.mean_inference_ms) / n;
            state.snapshot.source_frame_counter = frame.frame_counter;
            state.snapshot.hands = std::move(observations);
            state.snapshot.last_error.clear();
            state.snapshot.status_detail = state.snapshot.hands.empty()
                ? "Ready; no hand detected in the latest processed frame"
                : "Ready; latest local hand result published";
        } else {
            ++state.snapshot.inference_failures;
            state.snapshot.last_error = error;
            state.snapshot.status_detail = "Latest gesture inference failed";
            LOG_ERROR(kLogTag, error);
        }
        PublishSnapshotLocked(state);
    }

    state.backend->Shutdown();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.backend_ready = false;
    state.snapshot.worker_running = false;
    state.snapshot.worker_busy = false;
    state.snapshot.status_detail = "Interaction worker stopped";
    PublishSnapshotLocked(state);
}

void EnsureStarted() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.started) return;
    state.stop_requested = false;
    state.backend = CreatePhysicalMediaPipeHandGestureBackend();
    state.started = true;
    state.worker = std::thread(WorkerMain);
    state.cv.notify_all();
    LOG_DEBUG(kLogTag, "Physical interaction worker started");
}

} // namespace

void TickPhysicalInteraction() {
    EnsureStarted();
    auto& state = State();

    PhysicalFrameBus::FrameView frame;
    uint64_t last_seen = 0;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        last_seen = state.last_seen_frame_counter;
    }
    if (!PhysicalFrameBus::Instance().PullLatestFrameView(frame, last_seen)) return;

    std::lock_guard<std::mutex> lock(state.mutex);
    state.last_seen_frame_counter = last_seen;
    if (!state.config.enabled) return;

    const uint64_t capture_ns = frame.metadata.capture_steady_ns != 0
        ? frame.metadata.capture_steady_ns : SteadyNowNs();
    if (state.config.target_fps > 0.0 && state.last_submitted_capture_ns != 0) {
        const uint64_t interval_ns = static_cast<uint64_t>(
            1000000000.0 / std::max(1.0, state.config.target_fps));
        if (capture_ns > state.last_submitted_capture_ns &&
            capture_ns - state.last_submitted_capture_ns < interval_ns) {
            ++state.snapshot.frames_cadence_skipped;
            PublishSnapshotLocked(state);
            return;
        }
    }

    state.last_submitted_capture_ns = capture_ns;
    if (state.pending_frame) ++state.snapshot.frames_replaced;
    state.pending_frame = std::move(frame);
    ++state.snapshot.frames_submitted;
    PublishSnapshotLocked(state);
    state.cv.notify_one();
}

PhysicalHandGestureConfig GetPhysicalHandGestureConfig() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.config;
}

void RequestConfigurePhysicalHandGestures(
    const PhysicalHandGestureConfig& requested)
{
    EnsureStarted();
    auto config = requested;
    config.max_hands = std::clamp(config.max_hands, 1, 4);
    config.min_hand_detection_confidence =
        std::clamp(config.min_hand_detection_confidence, 0.0f, 1.0f);
    config.min_hand_presence_confidence =
        std::clamp(config.min_hand_presence_confidence, 0.0f, 1.0f);
    config.min_tracking_confidence =
        std::clamp(config.min_tracking_confidence, 0.0f, 1.0f);
    config.min_gesture_confidence =
        std::clamp(config.min_gesture_confidence, 0.0f, 1.0f);
    config.target_fps = std::clamp(config.target_fps, 1.0, 60.0);

    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = std::move(config);
    ++state.config_generation;
    state.backend_ready = false;
    state.pending_frame.reset();
    state.last_submitted_capture_ns = 0;
    state.snapshot.status_detail = "Gesture backend reconfiguration requested";
    PublishSnapshotLocked(state);
    state.cv.notify_all();
}

void RequestSetPhysicalHandGesturesEnabled(bool enabled) {
    auto config = GetPhysicalHandGestureConfig();
    config.enabled = enabled;
    RequestConfigurePhysicalHandGestures(config);
}

void ShutdownPhysicalInteraction() {
    auto& state = State();
    std::thread worker;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.started) return;
        state.stop_requested = true;
        state.pending_frame.reset();
        state.cv.notify_all();
        worker = std::move(state.worker);
    }
    if (worker.joinable()) worker.join();

    std::lock_guard<std::mutex> lock(state.mutex);
    state.backend.reset();
    state.started = false;
    state.stop_requested = false;
    state.backend_ready = false;
    state.last_seen_frame_counter = 0;
    state.last_submitted_capture_ns = 0;
    state.config_generation = 1;
    LOG_DEBUG(kLogTag, "Physical interaction worker shut down");
}

}}} // namespace GRIM::Perception::Physical
