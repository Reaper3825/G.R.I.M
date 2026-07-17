#pragma once

#include "PhysicalGestureControlResult.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

enum class PhysicalGestureAction : uint8_t {
    ControlArm = 0,
    ControlDisarm,
    PointerMove,
    MouseLeftClick,
    MouseRightClick,
    VoiceWake
};

enum class PhysicalGestureTrigger : uint8_t {
    Started = 0,
    Held,
    Released
};

struct PhysicalGestureBinding {
    std::string binding_id;
    std::string gesture_label;
    PhysicalGestureAction action = PhysicalGestureAction::VoiceWake;
    PhysicalGestureTrigger trigger = PhysicalGestureTrigger::Held;
    PhysicalHandedness handedness = PhysicalHandedness::Unknown; // either hand
    uint64_t minimum_hold_ms = 0;
    uint64_t cooldown_ms = 0;
    bool requires_armed = false;
    bool enabled = true;
    int priority = 0;
};

struct PhysicalGestureControlConfig {
    bool enabled = true;
    bool dry_run = false;
    PhysicalHandedness preferred_hand = PhysicalHandedness::Unknown; // either hand

    float enter_confidence = 0.75f;
    float exit_confidence = 0.55f;
    uint64_t activation_dwell_ms = 250;
    uint64_t release_grace_ms = 250;
    uint64_t result_stale_ms = 650;

    uint64_t armed_timeout_ms = 15000;

    float pointer_gain_pixels = 1600.0f;
    float pointer_smoothing = 0.45f; // Legacy persisted value; One Euro is used.
    float pointer_deadzone_normalized = 0.0008f;
    float pointer_deadzone_start_multiplier = 1.75f;
    float pointer_one_euro_min_cutoff = 1.2f;
    float pointer_one_euro_beta = 0.025f;
    float pointer_one_euro_derivative_cutoff = 1.0f;
    float pointer_outlier_velocity_normalized_per_second = 4.0f;
    float pointer_tip_weight = 0.70f;
    float pointer_hand_lock_radius_normalized = 0.24f;
    uint64_t pointer_hand_loss_grace_ms = 450;
    int max_pointer_step_pixels = 80;
    // Camera previews are mirrored for natural interaction, so horizontal
    // pointer deltas must be mirrored as well to follow the displayed hand.
    bool invert_pointer_x = true;
    bool invert_pointer_y = false;

    std::vector<PhysicalGestureBinding> bindings;
};

std::vector<PhysicalGestureBinding> DefaultPhysicalGestureBindings();
PhysicalGestureControlConfig DefaultPhysicalGestureControlConfig();

const char* PhysicalGestureActionId(PhysicalGestureAction action) noexcept;
const char* PhysicalGestureActionDisplayName(PhysicalGestureAction action) noexcept;
bool TryParsePhysicalGestureAction(const std::string& value,
                                   PhysicalGestureAction& action) noexcept;
const char* PhysicalGestureTriggerId(PhysicalGestureTrigger trigger) noexcept;
bool TryParsePhysicalGestureTrigger(const std::string& value,
                                    PhysicalGestureTrigger& trigger) noexcept;
std::vector<std::string> ValidatePhysicalGestureBindings(
    const PhysicalGestureControlConfig& config);

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
