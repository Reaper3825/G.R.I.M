#pragma once

#include "PhysicalHandGestureResult.hpp"

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

enum class PhysicalGesturePhase : uint8_t {
    Started = 0,
    Held,
    Released
};

struct PhysicalGestureEvent {
    uint64_t sequence = 0;
    uint64_t source_frame_counter = 0;
    uint64_t event_steady_ns = 0;
    PhysicalGesturePhase phase = PhysicalGesturePhase::Started;
    PhysicalHandedness handedness = PhysicalHandedness::Unknown;
    std::string gesture_label;
    float confidence = 0.0f;
    uint64_t held_ms = 0;

    // Pointer control position normalized to the raw image. The controller may
    // derive it from multiple landmarks without exposing backend types.
    float pointer_x = 0.0f;
    float pointer_y = 0.0f;
    bool has_pointer = false;
};

struct PhysicalGestureControlStatus {
    bool enabled = true;
    bool dry_run = false;
    bool armed = false;
    bool pointer_active = false;
    bool hand_lock_active = false;
    bool custom_cursor_active = false;
    std::string cursor_error;
    std::string candidate_gesture;
    std::string stable_gesture;
    std::string pointer_activation_gesture;
    PhysicalHandedness active_hand = PhysicalHandedness::Unknown;
    uint64_t last_source_frame_counter = 0;
    uint64_t armed_until_steady_ns = 0;
    uint64_t events_emitted = 0;
    uint64_t actions_executed = 0;
    uint64_t actions_previewed = 0;
    uint64_t actions_blocked = 0;
    uint64_t actions_failed = 0;
    uint64_t pointer_samples = 0;
    uint64_t pointer_moves_emitted = 0;
    uint64_t pointer_outliers_rejected = 0;
    uint64_t pointer_classifier_bypass_frames = 0;
    uint64_t hand_reacquisitions = 0;
    uint64_t pinch_clicks_emitted = 0;
    float pointer_raw_x = 0.0f;
    float pointer_raw_y = 0.0f;
    float pointer_filtered_x = 0.0f;
    float pointer_filtered_y = 0.0f;
    float pointer_sample_hz = 0.0f;
    bool pinch_tracking = false;
    bool pinch_closed = false;
    float pinch_distance_ratio = 0.0f;
    float pinch_visual_openness = 1.0f;
    std::string last_action;
    std::string last_block_reason;
    std::string last_error;
};

}}} // namespace GRIM::Perception::Physical
