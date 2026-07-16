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

    // Index-fingertip position (landmark 8), normalized to the raw image.
    // This remains useful for pointer routing without exposing backend types.
    float pointer_x = 0.0f;
    float pointer_y = 0.0f;
    bool has_pointer = false;
};

struct PhysicalGestureControlStatus {
    bool enabled = true;
    bool armed = false;
    bool pointer_active = false;
    std::string candidate_gesture;
    std::string stable_gesture;
    PhysicalHandedness active_hand = PhysicalHandedness::Unknown;
    uint64_t last_source_frame_counter = 0;
    uint64_t armed_until_steady_ns = 0;
    uint64_t events_emitted = 0;
    uint64_t actions_executed = 0;
    uint64_t actions_blocked = 0;
    uint64_t actions_failed = 0;
    std::string last_action;
    std::string last_block_reason;
    std::string last_error;
};

}}} // namespace GRIM::Perception::Physical
