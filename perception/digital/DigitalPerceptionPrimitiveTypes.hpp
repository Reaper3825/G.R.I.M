#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "DigitalCaptureTypes.hpp"

namespace GRIM { namespace Perception { namespace Digital {

enum class DigitalPrimitiveStatus : std::uint8_t {
    Ok = 0,
    Disabled,
    Unsupported,
    Unavailable,
    Failed
};

struct DigitalTextRegion {
    DigitalRect frame_rect{}; // Frame-local pixels, never desktop coordinates.
    std::string text;
    float confidence = 0.0f;  // [0, 1]
};

struct DigitalOcrResult {
    DigitalPrimitiveStatus status = DigitalPrimitiveStatus::Unavailable;
    std::string provider;
    std::string error;
    std::string full_text;
    std::vector<DigitalTextRegion> regions;
    float mean_confidence = 0.0f;
    double duration_ms = 0.0;
};

struct DigitalUiElement {
    DigitalRect desktop_rect{}; // Physical virtual-desktop pixels.
    std::string name;
    std::string automation_id;
    std::string role;
    bool enabled = false;
    bool offscreen = false;
    bool password = false;
};

struct DigitalAutomationResult {
    DigitalPrimitiveStatus status = DigitalPrimitiveStatus::Unsupported;
    std::string provider;
    std::string error;
    std::string target_window;
    std::vector<DigitalUiElement> elements;
    bool target_matches_capture = false;
    bool target_changed_since_capture = false;
    bool truncated = false;
    double duration_ms = 0.0;
};

struct DigitalPerceptionPrimitiveSnapshot {
    std::uint64_t source_frame_counter = 0;
    std::uint64_t source_capture_wall_ns = 0;
    std::uint64_t analyzed_steady_ns = 0;
    DigitalCaptureMetadata source_metadata{};
    DigitalOcrResult ocr{};
    DigitalAutomationResult automation{};
};

const char* ToString(DigitalPrimitiveStatus status) noexcept;

}}} // namespace GRIM::Perception::Digital
