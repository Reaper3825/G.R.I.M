#include "DigitalEnvironmentLoop.hpp"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

#include "DigitalCaptureSource.hpp"
#include "DigitalFrameBus.hpp"
#include "logger.hpp"

namespace GRIM { namespace Perception { namespace Digital {

namespace {

constexpr const char* kLogTag = "DigitalEnvironment";

struct State {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool running = false;
    bool stop_requested = false;
    bool capture_requested = false;
    std::uint64_t next_frame_counter = 1;
    DigitalEnvironmentConfig config{};
    DigitalEnvironmentStatus status{};
};

State& GetState() {
    static State state;
    return state;
}

void CaptureThread() {
    auto source = CreatePlatformDigitalCaptureSource();
    auto& state = GetState();

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.status.backend = source ? source->BackendName() : "unavailable";
    }

    while (true) {
        DigitalEnvironmentConfig config;
        std::uint64_t frame_counter = 0;
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            if (state.stop_requested) break;
            config = state.config;
            if (state.next_frame_counter == 0) state.next_frame_counter = 1;
            frame_counter = state.next_frame_counter++;
            state.capture_requested = false;
        }

        DigitalCaptureResult result;
        try {
            if (source) {
                result = source->Capture(config.request);
            } else {
                result.metadata.status = DigitalCaptureStatus::Unsupported;
                result.metadata.error = "platform capture source factory returned null";
                result.metadata.backend = "unavailable";
                result.metadata.mode = config.request.mode;
            }
        } catch (const std::exception& e) {
            result.metadata.status = DigitalCaptureStatus::CaptureFailed;
            result.metadata.error = std::string("capture backend threw: ") + e.what();
            result.metadata.backend = source ? source->BackendName() : "unavailable";
            result.metadata.mode = config.request.mode;
        } catch (...) {
            result.metadata.status = DigitalCaptureStatus::CaptureFailed;
            result.metadata.error = "capture backend threw a non-standard exception";
            result.metadata.backend = source ? source->BackendName() : "unavailable";
            result.metadata.mode = config.request.mode;
        }

        DigitalFrameBus::Instance().Publish(result, frame_counter);

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            ++state.status.capture_attempts;
            state.status.last_frame_counter = frame_counter;
            state.status.last_status = result.metadata.status;
            state.status.last_error = result.metadata.error;
            state.status.last_capture_duration_ms = result.metadata.capture_duration_ms;
            state.status.backend = result.metadata.backend;
            if (result.Succeeded()) {
                ++state.status.successful_captures;
            } else {
                ++state.status.failed_captures;
            }
        }

        if (!result.Succeeded()) {
            LOG_ERROR(kLogTag, std::string("capture attempt failed: status=") +
                      ToString(result.metadata.status) + " error=" + result.metadata.error);
        }

        std::unique_lock<std::mutex> lock(state.mutex);
        const auto interval = std::max(config.capture_interval,
                                       std::chrono::milliseconds(50));
        state.wake.wait_for(lock, interval, [&state] {
            return state.stop_requested || state.capture_requested;
        });
        if (state.stop_requested) break;
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    state.running = false;
    state.status.running = false;
}

} // namespace

void StartDigitalEnvironment(const DigitalEnvironmentConfig& config) {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.running) {
        state.config = config;
        state.capture_requested = true;
        state.wake.notify_all();
        return;
    }

    state.config = config;
    state.stop_requested = false;
    state.capture_requested = true;
    state.status = DigitalEnvironmentStatus{};
    state.status.running = true;
    state.running = true;
    DigitalFrameBus::Instance().Reset();
    state.worker = std::thread(CaptureThread);
    LOG_DEBUG(kLogTag, "digital capture worker started");
}

void RequestDigitalCaptureNow() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.running) return;
    state.capture_requested = true;
    state.wake.notify_all();
}

void UpdateDigitalEnvironmentConfig(const DigitalEnvironmentConfig& config) {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = config;
    if (state.running) {
        state.capture_requested = true;
        state.wake.notify_all();
    }
}

void ShutdownDigitalEnvironment() {
    auto& state = GetState();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.running && !state.worker.joinable()) return;
        state.stop_requested = true;
        state.wake.notify_all();
    }
    if (state.worker.joinable()) state.worker.join();

    std::lock_guard<std::mutex> lock(state.mutex);
    state.running = false;
    state.status.running = false;
    DigitalFrameBus::Instance().Reset();
    LOG_DEBUG(kLogTag, "digital capture worker stopped");
}

bool IsDigitalEnvironmentRunning() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.running;
}

DigitalEnvironmentConfig GetDigitalEnvironmentConfigSnapshot() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.config;
}

DigitalEnvironmentStatus GetDigitalEnvironmentStatusSnapshot() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    DigitalEnvironmentStatus result = state.status;
    result.running = state.running;
    return result;
}

}}} // namespace GRIM::Perception::Digital
