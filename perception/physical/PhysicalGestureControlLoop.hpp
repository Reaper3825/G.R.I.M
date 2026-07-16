#pragma once

#include "PhysicalGestureControlResult.hpp"

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalGestureControlConfig {
    bool enabled = true;
    PhysicalHandedness preferred_hand = PhysicalHandedness::Unknown; // either hand

    float enter_confidence = 0.75f;
    float exit_confidence = 0.55f;
    uint64_t activation_dwell_ms = 250;
    uint64_t release_grace_ms = 250;
    uint64_t result_stale_ms = 650;

    std::string arm_gesture = "Victory";
    uint64_t arm_hold_ms = 800;
    std::string disarm_gesture = "Open_Palm";
    uint64_t disarm_hold_ms = 700;
    uint64_t armed_timeout_ms = 15000;

    std::string pointer_gesture = "Pointing_Up";
    std::string left_click_gesture = "Closed_Fist";
    std::string right_click_gesture = "Thumb_Down";
    float pointer_gain_pixels = 1600.0f;
    float pointer_smoothing = 0.45f;
    float pointer_deadzone_normalized = 0.0025f;
    int max_pointer_step_pixels = 55;
    bool invert_pointer_x = false;
    bool invert_pointer_y = false;
    uint64_t click_cooldown_ms = 650;

    std::string wake_gesture = "ILoveYou";
    uint64_t wake_hold_ms = 1100;
    uint64_t wake_cooldown_ms = 10000;
    bool wake_requires_armed = false;
};

// Main-thread controller tick. It consumes new inference results, emits
// stabilized events, and executes only the explicitly mapped local actions.
void TickPhysicalGestureControl();
void ShutdownPhysicalGestureControl();

PhysicalGestureControlConfig GetPhysicalGestureControlConfig();
PhysicalGestureControlStatus GetPhysicalGestureControlStatus();
void RequestConfigurePhysicalGestureControl(
    const PhysicalGestureControlConfig& config);
void RequestSetPhysicalGestureControlEnabled(bool enabled);

}}} // namespace GRIM::Perception::Physical
