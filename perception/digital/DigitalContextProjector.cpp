#include "DigitalContextProjector.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <sstream>
#include <string>

#include "DigitalFrameBus.hpp"
#include "DigitalPerceptionPrimitiveBus.hpp"
#include "MMO/Core/SessionContextManager.hpp"

namespace GRIM { namespace Perception { namespace Digital {

namespace {

constexpr const char* kDefaultSession = "default";

struct ProjectorState {
    std::mutex mutex;
    std::uint64_t last_seen_frame = 0;
    std::uint64_t last_seen_primitives = 0;
    ::GRIM::MMO::VisualContext::DigitalVisual projection{};
};

ProjectorState& State() {
    static ProjectorState state;
    return state;
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string ClassifyScene(const DigitalCaptureMetadata& metadata) {
    const std::string process = Lower(metadata.active_process_name);
    if (process.find("chrome") != std::string::npos ||
        process.find("firefox") != std::string::npos ||
        process.find("msedge") != std::string::npos ||
        process.find("brave") != std::string::npos) return "web-browser";
    if (process == "code.exe" || process.find("devenv") != std::string::npos ||
        process.find("idea") != std::string::npos ||
        process.find("pycharm") != std::string::npos) return "ide-code";
    if (process.find("terminal") != std::string::npos ||
        process.find("powershell") != std::string::npos ||
        process == "cmd.exe" || process == "windowsterminal.exe") return "terminal";
    if (process.find("discord") != std::string::npos ||
        process.find("slack") != std::string::npos ||
        process.find("teams") != std::string::npos) return "chat-messaging";
    if (process.find("winword") != std::string::npos ||
        process.find("excel") != std::string::npos ||
        process.find("notepad") != std::string::npos) return "document";
    return process.empty() ? "unknown" : "desktop-application";
}

} // namespace

void TickDigitalContextProjector() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);

    bool changed = false;
    DigitalFrameBus::FrameView view;
    if (DigitalFrameBus::Instance().PullLatest(view, state.last_seen_frame)) {
        state.projection.active_window = view.metadata.active_window_title;
        state.projection.active_process = view.metadata.active_process_name;
        state.projection.scene_classification = ClassifyScene(view.metadata);
        state.projection.monitor_id = view.metadata.monitor_id;
        state.projection.source_device_id = view.metadata.source_device_id;
        state.projection.source_platform = view.metadata.source_platform;
        state.projection.source_transport = view.metadata.source_transport;
        state.projection.capture_status = ToString(view.metadata.status);
        state.projection.capture_error = view.metadata.error;
        state.projection.capture_backend = view.metadata.backend;
        state.projection.provenance_frame_counter = view.frame_counter;
        state.projection.capture_wall_ns = view.metadata.capture_wall_ns;
        changed = true;
    }

    DigitalPerceptionPrimitiveBus::SnapshotView primitive_view;
    if (DigitalPerceptionPrimitiveBus::Instance().PullLatest(
            primitive_view, state.last_seen_primitives) && primitive_view.snapshot) {
        const auto& primitives = *primitive_view.snapshot;
        state.projection.ocr_text = primitives.ocr.full_text.substr(0, 16 * 1024);
        state.projection.ocr_status = ToString(primitives.ocr.status);
        state.projection.ocr_error = primitives.ocr.error;
        state.projection.ocr_provider = primitives.ocr.provider;
        state.projection.ocr_mean_confidence = primitives.ocr.mean_confidence;
        state.projection.ocr_regions.clear();
        const std::size_t ocr_limit = std::min<std::size_t>(primitives.ocr.regions.size(), 64);
        state.projection.ocr_regions.reserve(ocr_limit);
        for (std::size_t i = 0; i < ocr_limit; ++i) {
            const auto& region = primitives.ocr.regions[i];
            std::ostringstream line;
            line << region.text << " @ " << region.frame_rect.x << ','
                 << region.frame_rect.y << ' ' << region.frame_rect.width << 'x'
                 << region.frame_rect.height;
            state.projection.ocr_regions.push_back(line.str());
        }

        state.projection.automation_status = ToString(primitives.automation.status);
        state.projection.automation_error = primitives.automation.error;
        state.projection.automation_provider = primitives.automation.provider;
        state.projection.automation_target_window = primitives.automation.target_window;
        state.projection.automation_target_matches_capture =
            primitives.automation.target_matches_capture;
        state.projection.automation_target_changed =
            primitives.automation.target_changed_since_capture;
        state.projection.preferred_grounding_source =
            primitives.automation.status == DigitalPrimitiveStatus::Ok &&
                    !primitives.automation.elements.empty() &&
                    primitives.automation.target_matches_capture
                ? primitives.automation.provider
                : primitives.ocr.provider;
        state.projection.ui_elements.clear();
        const std::size_t ui_limit =
            std::min<std::size_t>(primitives.automation.elements.size(), 96);
        state.projection.ui_elements.reserve(ui_limit);
        for (std::size_t i = 0; i < ui_limit; ++i) {
            const auto& element = primitives.automation.elements[i];
            std::ostringstream line;
            line << element.role;
            if (!element.name.empty()) line << " '" << element.name << '\'';
            line << " @ " << element.desktop_rect.x << ',' << element.desktop_rect.y
                 << ' ' << element.desktop_rect.width << 'x' << element.desktop_rect.height;
            state.projection.ui_elements.push_back(line.str());
        }
        state.projection.primitive_provenance_frame_counter =
            primitives.source_frame_counter;
        changed = true;
    }

    if (!changed) return;

    ::GRIM::MMO::SessionContextManager::instance().updateDigitalVisual(
        kDefaultSession, state.projection);
}

void ShutdownDigitalContextProjector() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.last_seen_frame = 0;
    state.last_seen_primitives = 0;
    state.projection = {};
}

}}} // namespace GRIM::Perception::Digital
