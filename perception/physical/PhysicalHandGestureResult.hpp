#pragma once

#include "PhysicalFrameBus.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// MediaPipe-free data contract for the human-interaction branch. Keeping the
// public result free of backend types lets the UI and future controller consume
// gestures without linking to MediaPipe.
enum class PhysicalHandGestureBackendState : uint8_t {
    Disabled = 0,
    BackendUnavailable,
    ModelMissing,
    Initializing,
    Ready,
    Failed
};

enum class PhysicalHandedness : uint8_t {
    Unknown = 0,
    Left,
    Right
};

struct PhysicalHandLandmark {
    float normalized_x = 0.0f;
    float normalized_y = 0.0f;
    float normalized_z = 0.0f;
    float raw_pixel_x  = 0.0f;
    float raw_pixel_y  = 0.0f;
    float world_x_m    = 0.0f;
    float world_y_m    = 0.0f;
    float world_z_m    = 0.0f;
    bool  has_world    = false;
};

struct PhysicalHandObservation {
    uint64_t source_frame_counter     = 0;
    uint64_t source_capture_steady_ns = 0;
    uint64_t result_steady_ns         = 0;
    int      raw_image_width          = 0;
    int      raw_image_height         = 0;

    PhysicalHandedness handedness            = PhysicalHandedness::Unknown;
    float              handedness_confidence = 0.0f;
    std::string        gesture_label;
    float              gesture_confidence    = 0.0f;
    std::array<PhysicalHandLandmark, 21> landmarks{};
    uint32_t landmark_count = 0;
};

// Operational state and the latest result are deliberately published together.
// This makes unavailable/error/idle states visible even before a camera frame
// has been processed.
struct PhysicalHandGestureSnapshot {
    uint64_t publish_sequence     = 0;
    uint64_t source_frame_counter = 0;
    uint64_t published_steady_ns  = 0;

    // Pin the exact immutable frame consumed by the interaction worker. The
    // camera bus is a latest-frame slot, so an asynchronous MediaPipe result
    // will almost never match the frame in that slot by the time inference
    // completes. Keeping the source view with the result lets preview clients
    // render a coherent frame/landmark pair without copying image pixels.
    PhysicalFrameBus::FrameView source_frame;

    PhysicalHandGestureBackendState backend_state =
        PhysicalHandGestureBackendState::BackendUnavailable;
    std::string backend_name = "MediaPipe Tasks C";
    std::string backend_version;
    std::string model_path;
    std::string status_detail;
    std::string last_error;

    bool enabled                = true;
    bool offline_only           = true;
    bool telemetry_disabled     = false;
    bool worker_running         = false;
    bool worker_busy            = false;

    uint64_t frames_submitted       = 0;
    uint64_t frames_processed       = 0;
    uint64_t frames_replaced        = 0;
    uint64_t frames_cadence_skipped = 0;
    uint64_t inference_failures     = 0;

    double last_inference_ms = 0.0;
    double mean_inference_ms = 0.0;
    double p95_inference_ms = 0.0;
    double configured_target_fps = 30.0;
    double effective_target_fps = 30.0;

    std::vector<PhysicalHandObservation> hands;
};

}}} // namespace GRIM::Perception::Physical
