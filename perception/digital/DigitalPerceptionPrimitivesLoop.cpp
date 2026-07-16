#include "DigitalPerceptionPrimitivesLoop.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

#include "DigitalAutomationProvider.hpp"
#include "DigitalFrameBus.hpp"
#include "DigitalOcrProvider.hpp"
#include "DigitalPerceptionPrimitiveBus.hpp"
#include "logger.hpp"

namespace GRIM { namespace Perception { namespace Digital {

namespace {

constexpr const char* kLogTag = "DigitalPrimitives";

struct State {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool running = false;
    bool stop_requested = false;
    DigitalPerceptionPrimitivesConfig config{};
    DigitalPerceptionPrimitivesStatus status{};
};

State& GetState() {
    static State state;
    return state;
}

std::uint64_t SteadyNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

DigitalOcrResult SkippedOcr(const char* provider,
                            DigitalPrimitiveStatus status,
                            std::string error = {}) {
    DigitalOcrResult result;
    result.provider = provider ? provider : "unavailable";
    result.status = status;
    result.error = std::move(error);
    return result;
}

DigitalAutomationResult SkippedAutomation(const char* provider,
                                           DigitalPrimitiveStatus status,
                                           std::string error = {}) {
    DigitalAutomationResult result;
    result.provider = provider ? provider : "unavailable";
    result.status = status;
    result.error = std::move(error);
    return result;
}

void WorkerThread() {
    auto ocr = CreateDefaultDigitalOcrProvider();
    auto automation = CreatePlatformDigitalAutomationProvider();
    auto& state = GetState();
    std::uint64_t frame_cursor = 0;

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.status.ocr_provider = ocr ? ocr->ProviderName() : "unavailable";
        state.status.automation_provider =
            automation ? automation->ProviderName() : "unavailable";
    }

    while (true) {
        {
            std::unique_lock<std::mutex> lock(state.mutex);
            if (state.stop_requested) break;
        }

        DigitalFrameBus::FrameView frame;
        if (!DigitalFrameBus::Instance().PullLatest(frame, frame_cursor)) {
            std::unique_lock<std::mutex> lock(state.mutex);
            state.wake.wait_for(lock, std::chrono::milliseconds(50), [&state] {
                return state.stop_requested;
            });
            if (state.stop_requested) break;
            continue;
        }

        DigitalPerceptionPrimitivesConfig config;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            config = state.config;
        }

        DigitalPerceptionPrimitiveSnapshot snapshot;
        snapshot.source_frame_counter = frame.frame_counter;
        snapshot.source_capture_wall_ns = frame.metadata.capture_wall_ns;
        snapshot.source_metadata = frame.metadata;

        if (frame.metadata.status != DigitalCaptureStatus::Ok || frame.image.empty()) {
            const std::string reason = "source capture unavailable: " +
                std::string(ToString(frame.metadata.status)) +
                (frame.metadata.error.empty() ? std::string{} : " - " + frame.metadata.error);
            snapshot.ocr = SkippedOcr(ocr ? ocr->ProviderName() : nullptr,
                                      DigitalPrimitiveStatus::Unavailable, reason);
            snapshot.automation = SkippedAutomation(
                automation ? automation->ProviderName() : nullptr,
                DigitalPrimitiveStatus::Unavailable, reason);
        } else {
            // Native automation is sampled first so its foreground target is as
            // close as possible to the captured metadata. It remains optional;
            // OCR always analyzes the exact immutable frame independently.
            const bool local_windows_source =
                frame.metadata.source_device_id == "local" &&
                frame.metadata.source_platform == "windows" &&
                frame.metadata.source_transport == "native";
            if (!local_windows_source) {
                snapshot.automation = SkippedAutomation(
                    automation ? automation->ProviderName() : nullptr,
                    DigitalPrimitiveStatus::Unsupported,
                    "host-native automation cannot inspect a remote or non-Windows frame");
            } else if (!config.automation_enabled) {
                snapshot.automation = SkippedAutomation(
                    automation ? automation->ProviderName() : nullptr,
                    DigitalPrimitiveStatus::Disabled);
            } else if (!automation) {
                snapshot.automation = SkippedAutomation(
                    nullptr, DigitalPrimitiveStatus::Unavailable,
                    "platform automation provider factory returned null");
            } else {
                try {
                    snapshot.automation = automation->InspectForeground(
                        frame.metadata,
                        std::max<std::size_t>(1, config.max_automation_elements));
                } catch (const std::exception& e) {
                    snapshot.automation = SkippedAutomation(
                        automation->ProviderName(), DigitalPrimitiveStatus::Failed,
                        std::string("automation provider threw: ") + e.what());
                } catch (...) {
                    snapshot.automation = SkippedAutomation(
                        automation->ProviderName(), DigitalPrimitiveStatus::Failed,
                        "automation provider threw a non-standard exception");
                }
            }

            if (!config.ocr_enabled) {
                snapshot.ocr = SkippedOcr(ocr ? ocr->ProviderName() : nullptr,
                                          DigitalPrimitiveStatus::Disabled);
            } else if (!ocr) {
                snapshot.ocr = SkippedOcr(
                    nullptr, DigitalPrimitiveStatus::Unavailable,
                    "OCR provider factory returned null");
            } else {
                try {
                    snapshot.ocr = ocr->Recognize(
                        frame.image, frame.metadata.dpi_x, frame.metadata.dpi_y);
                } catch (const std::exception& e) {
                    snapshot.ocr = SkippedOcr(
                        ocr->ProviderName(), DigitalPrimitiveStatus::Failed,
                        std::string("OCR provider threw: ") + e.what());
                } catch (...) {
                    snapshot.ocr = SkippedOcr(
                        ocr->ProviderName(), DigitalPrimitiveStatus::Failed,
                        "OCR provider threw a non-standard exception");
                }
            }
        }

        snapshot.analyzed_steady_ns = SteadyNowNs();
        try {
            DigitalPerceptionPrimitiveBus::Instance().Publish(snapshot);
        } catch (const std::exception& e) {
            LOG_ERROR(kLogTag, std::string("failed to publish primitive snapshot: ") + e.what());
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            ++state.status.processed_frames;
            state.status.last_source_frame_counter = frame.frame_counter;
            state.status.last_ocr_duration_ms = snapshot.ocr.duration_ms;
            state.status.last_automation_duration_ms = snapshot.automation.duration_ms;
            if (snapshot.ocr.status == DigitalPrimitiveStatus::Failed) {
                state.status.last_error = snapshot.ocr.error;
            } else if (snapshot.automation.status == DigitalPrimitiveStatus::Failed) {
                state.status.last_error = snapshot.automation.error;
            } else {
                state.status.last_error.clear();
            }
        }
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    state.running = false;
    state.status.running = false;
}

} // namespace

void StartDigitalPerceptionPrimitives(const DigitalPerceptionPrimitivesConfig& config) {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.running) {
        state.config = config;
        return;
    }
    state.config = config;
    state.stop_requested = false;
    state.status = DigitalPerceptionPrimitivesStatus{};
    state.status.running = true;
    state.running = true;
    DigitalPerceptionPrimitiveBus::Instance().Reset();
    state.worker = std::thread(WorkerThread);
    LOG_DEBUG(kLogTag, "digital perception primitives worker started");
}

void UpdateDigitalPerceptionPrimitivesConfig(
    const DigitalPerceptionPrimitivesConfig& config) {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = config;
}

void ShutdownDigitalPerceptionPrimitives() {
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
    DigitalPerceptionPrimitiveBus::Instance().Reset();
    LOG_DEBUG(kLogTag, "digital perception primitives worker stopped");
}

bool IsDigitalPerceptionPrimitivesRunning() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.running;
}

DigitalPerceptionPrimitivesConfig GetDigitalPerceptionPrimitivesConfigSnapshot() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.config;
}

DigitalPerceptionPrimitivesStatus GetDigitalPerceptionPrimitivesStatusSnapshot() {
    auto& state = GetState();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto status = state.status;
    status.running = state.running;
    return status;
}

}}} // namespace GRIM::Perception::Digital
