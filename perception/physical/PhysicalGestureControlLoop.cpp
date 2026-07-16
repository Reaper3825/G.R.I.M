#include "PhysicalGestureControlLoop.hpp"

#include "PhysicalGestureEventBus.hpp"
#include "PhysicalHandGestureBus.hpp"
#include "core/platform_input.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

constexpr const char* kLogTag = "PHYSICAL_GESTURE_CONTROL";

uint64_t SteadyNowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t MillisecondsToNs(uint64_t milliseconds) {
    return milliseconds * 1000000ULL;
}

std::string UsableGestureLabel(const std::string& label) {
    if (label.empty() || label == "None" || label == "none") return {};
    return label;
}

enum class PendingActionType : uint8_t {
    MovePointer = 0,
    LeftClick,
    RightClick,
    WakeVoice
};

struct PendingAction {
    PendingActionType type = PendingActionType::MovePointer;
    int dx = 0;
    int dy = 0;
    std::string description;
};

struct PhysicalGestureControlState {
    std::mutex mutex;
    PhysicalGestureControlConfig config{};
    PhysicalGestureControlStatus status{};

    uint64_t last_hand_bus_sequence = 0;
    uint64_t last_processed_source_frame = 0;
    uint64_t last_result_seen_ns = 0;

    std::string candidate_gesture;
    uint64_t candidate_since_ns = 0;
    float candidate_confidence = 0.0f;
    PhysicalHandedness candidate_hand = PhysicalHandedness::Unknown;
    float candidate_pointer_x = 0.0f;
    float candidate_pointer_y = 0.0f;
    bool candidate_has_pointer = false;

    std::string stable_gesture;
    uint64_t stable_since_ns = 0;
    uint64_t last_stable_confident_ns = 0;
    float stable_confidence = 0.0f;
    PhysicalHandedness stable_hand = PhysicalHandedness::Unknown;
    float stable_pointer_x = 0.0f;
    float stable_pointer_y = 0.0f;
    bool stable_has_pointer = false;

    bool arm_latched = false;
    bool disarm_latched = false;
    bool wake_latched = false;
    uint64_t last_click_ns = 0;
    uint64_t last_wake_ns = 0;

    float previous_pointer_x = 0.0f;
    float previous_pointer_y = 0.0f;
    bool have_previous_pointer = false;
    float smoothed_dx = 0.0f;
    float smoothed_dy = 0.0f;
    std::string last_logged_failure_action;
};

PhysicalGestureControlState& State() {
    static PhysicalGestureControlState state;
    return state;
}

void ResetPointerLocked(PhysicalGestureControlState& state) {
    state.status.pointer_active = false;
    state.have_previous_pointer = false;
    state.smoothed_dx = 0.0f;
    state.smoothed_dy = 0.0f;
}

void DisarmLocked(PhysicalGestureControlState& state,
                  const std::string& reason)
{
    state.status.armed = false;
    state.status.armed_until_steady_ns = 0;
    state.status.last_action = "control_disarmed";
    state.status.last_block_reason = reason;
    ResetPointerLocked(state);
    LOG_DEBUG(kLogTag, "Gesture control disarmed: " + reason);
}

void ExtendArmedWindowLocked(PhysicalGestureControlState& state,
                             uint64_t now_ns)
{
    state.status.armed_until_steady_ns =
        now_ns + MillisecondsToNs(state.config.armed_timeout_ms);
}

void RecordBlockedLocked(PhysicalGestureControlState& state,
                         const std::string& reason)
{
    ++state.status.actions_blocked;
    state.status.last_block_reason = reason;
}

void RouteEventLocked(PhysicalGestureControlState& state,
                      const PhysicalGestureEvent& event,
                      std::vector<PendingAction>& actions)
{
    const auto& config = state.config;
    const uint64_t now_ns = event.event_steady_ns;

    if (event.phase == PhysicalGesturePhase::Released) {
        if (event.gesture_label == config.arm_gesture) state.arm_latched = false;
        if (event.gesture_label == config.disarm_gesture) state.disarm_latched = false;
        if (event.gesture_label == config.wake_gesture) state.wake_latched = false;
        if (event.gesture_label == config.pointer_gesture) ResetPointerLocked(state);
        return;
    }

    if (event.gesture_label == config.arm_gesture &&
        event.held_ms >= config.arm_hold_ms && !state.arm_latched) {
        state.arm_latched = true;
        state.status.armed = true;
        state.status.last_action = "control_armed";
        state.status.last_block_reason.clear();
        ++state.status.actions_executed;
        ExtendArmedWindowLocked(state, now_ns);
        LOG_DEBUG(kLogTag, "Gesture control armed");
        return;
    }

    if (event.gesture_label == config.disarm_gesture &&
        event.held_ms >= config.disarm_hold_ms && !state.disarm_latched) {
        state.disarm_latched = true;
        ++state.status.actions_executed;
        DisarmLocked(state, "explicit disarm gesture");
        return;
    }

    if (event.gesture_label == config.wake_gesture &&
        event.held_ms >= config.wake_hold_ms && !state.wake_latched) {
        state.wake_latched = true;
        if (config.wake_requires_armed && !state.status.armed) {
            RecordBlockedLocked(state, "wake gesture requires armed control");
        } else if (state.last_wake_ns != 0 &&
                   now_ns - state.last_wake_ns <
                       MillisecondsToNs(config.wake_cooldown_ms)) {
            RecordBlockedLocked(state, "wake gesture is cooling down");
        } else {
            state.last_wake_ns = now_ns;
            actions.push_back({PendingActionType::WakeVoice, 0, 0,
                               "wake_voice"});
        }
        return;
    }

    const bool computer_action =
        event.gesture_label == config.pointer_gesture ||
        event.gesture_label == config.left_click_gesture ||
        event.gesture_label == config.right_click_gesture;
    if (computer_action && !state.status.armed) {
        if (event.phase == PhysicalGesturePhase::Started)
            RecordBlockedLocked(state, "computer control is not armed");
        return;
    }

    if (event.gesture_label == config.pointer_gesture && event.has_pointer) {
        ExtendArmedWindowLocked(state, now_ns);
        if (event.phase == PhysicalGesturePhase::Started ||
            !state.have_previous_pointer) {
            state.previous_pointer_x = event.pointer_x;
            state.previous_pointer_y = event.pointer_y;
            state.have_previous_pointer = true;
            state.status.pointer_active = true;
            return;
        }

        float nx = event.pointer_x - state.previous_pointer_x;
        float ny = event.pointer_y - state.previous_pointer_y;
        state.previous_pointer_x = event.pointer_x;
        state.previous_pointer_y = event.pointer_y;
        if (config.invert_pointer_x) nx = -nx;
        if (config.invert_pointer_y) ny = -ny;
        if (std::abs(nx) < config.pointer_deadzone_normalized) nx = 0.0f;
        if (std::abs(ny) < config.pointer_deadzone_normalized) ny = 0.0f;

        const float raw_dx = nx * config.pointer_gain_pixels;
        const float raw_dy = ny * config.pointer_gain_pixels;
        const float alpha = std::clamp(config.pointer_smoothing, 0.0f, 1.0f);
        state.smoothed_dx = alpha * raw_dx + (1.0f - alpha) * state.smoothed_dx;
        state.smoothed_dy = alpha * raw_dy + (1.0f - alpha) * state.smoothed_dy;
        const int limit = std::max(1, config.max_pointer_step_pixels);
        const int dx = std::clamp(static_cast<int>(std::lround(state.smoothed_dx)),
                                  -limit, limit);
        const int dy = std::clamp(static_cast<int>(std::lround(state.smoothed_dy)),
                                  -limit, limit);
        if (dx != 0 || dy != 0) {
            actions.push_back({PendingActionType::MovePointer, dx, dy,
                               "pointer_move"});
        }
        return;
    }

    if (event.phase != PhysicalGesturePhase::Started) return;
    if (event.gesture_label != config.left_click_gesture &&
        event.gesture_label != config.right_click_gesture) return;

    if (state.last_click_ns != 0 &&
        now_ns - state.last_click_ns < MillisecondsToNs(config.click_cooldown_ms)) {
        RecordBlockedLocked(state, "mouse click is cooling down");
        return;
    }
    state.last_click_ns = now_ns;
    ExtendArmedWindowLocked(state, now_ns);
    ResetPointerLocked(state);
    if (event.gesture_label == config.left_click_gesture) {
        actions.push_back({PendingActionType::LeftClick, 0, 0, "mouse_left_click"});
    } else {
        actions.push_back({PendingActionType::RightClick, 0, 0, "mouse_right_click"});
    }
}

PhysicalGestureEvent MakeEventLocked(
    const PhysicalGestureControlState& state,
    PhysicalGesturePhase phase,
    uint64_t now_ns)
{
    PhysicalGestureEvent event;
    event.source_frame_counter = state.status.last_source_frame_counter;
    event.event_steady_ns = now_ns;
    event.phase = phase;
    event.handedness = state.stable_hand;
    event.gesture_label = state.stable_gesture;
    event.confidence = state.stable_confidence;
    event.held_ms = state.stable_since_ns != 0 && now_ns >= state.stable_since_ns
        ? (now_ns - state.stable_since_ns) / 1000000ULL : 0;
    event.pointer_x = state.stable_pointer_x;
    event.pointer_y = state.stable_pointer_y;
    event.has_pointer = state.stable_has_pointer;
    return event;
}

void PublishAndRouteLocked(PhysicalGestureControlState& state,
                           PhysicalGesturePhase phase,
                           uint64_t now_ns,
                           std::vector<PendingAction>& actions)
{
    if (state.stable_gesture.empty()) return;
    auto event = PhysicalGestureEventBus::Instance()
        .PublishPhysicalGestureEvent(MakeEventLocked(state, phase, now_ns));
    ++state.status.events_emitted;
    RouteEventLocked(state, event, actions);
}

void ReleaseStableLocked(PhysicalGestureControlState& state,
                         uint64_t now_ns,
                         std::vector<PendingAction>& actions)
{
    if (state.stable_gesture.empty()) return;
    PublishAndRouteLocked(state, PhysicalGesturePhase::Released, now_ns, actions);
    state.stable_gesture.clear();
    state.stable_since_ns = 0;
    state.last_stable_confident_ns = 0;
    state.stable_confidence = 0.0f;
    state.stable_hand = PhysicalHandedness::Unknown;
    state.stable_has_pointer = false;
    state.status.stable_gesture.clear();
    state.status.active_hand = PhysicalHandedness::Unknown;
}

const PhysicalHandObservation* SelectControlHand(
    const PhysicalGestureControlConfig& config,
    const PhysicalHandGestureSnapshot& snapshot)
{
    const PhysicalHandObservation* best = nullptr;
    for (const auto& hand : snapshot.hands) {
        if (config.preferred_hand != PhysicalHandedness::Unknown &&
            hand.handedness != config.preferred_hand) continue;
        if (!best || hand.gesture_confidence > best->gesture_confidence)
            best = &hand;
    }
    return best;
}

void ProcessHandSnapshotLocked(PhysicalGestureControlState& state,
                               const PhysicalHandGestureSnapshot& snapshot,
                               uint64_t now_ns,
                               std::vector<PendingAction>& actions)
{
    state.status.last_source_frame_counter = snapshot.source_frame_counter;
    state.last_result_seen_ns = now_ns;
    const auto* hand = SelectControlHand(state.config, snapshot);
    const std::string label = hand
        ? UsableGestureLabel(hand->gesture_label) : std::string{};
    const float confidence = hand ? hand->gesture_confidence : 0.0f;
    const bool has_pointer = hand && hand->landmark_count > 8;
    const float pointer_x = has_pointer ? hand->landmarks[8].normalized_x : 0.0f;
    const float pointer_y = has_pointer ? hand->landmarks[8].normalized_y : 0.0f;

    if (label != state.candidate_gesture ||
        (hand && hand->handedness != state.candidate_hand)) {
        state.candidate_gesture = label;
        state.candidate_since_ns = now_ns;
        state.candidate_hand = hand ? hand->handedness : PhysicalHandedness::Unknown;
    }
    state.candidate_confidence = confidence;
    state.candidate_pointer_x = pointer_x;
    state.candidate_pointer_y = pointer_y;
    state.candidate_has_pointer = has_pointer;
    state.status.candidate_gesture = label;

    if (!state.stable_gesture.empty() && label == state.stable_gesture &&
        (!hand || hand->handedness == state.stable_hand)) {
        state.stable_confidence = confidence;
        state.stable_pointer_x = pointer_x;
        state.stable_pointer_y = pointer_y;
        state.stable_has_pointer = has_pointer;
        if (confidence >= state.config.exit_confidence) {
            state.last_stable_confident_ns = now_ns;
            PublishAndRouteLocked(state, PhysicalGesturePhase::Held,
                                  now_ns, actions);
        }
        return;
    }

    if (!label.empty() && confidence >= state.config.enter_confidence &&
        now_ns - state.candidate_since_ns >=
            MillisecondsToNs(state.config.activation_dwell_ms)) {
        ReleaseStableLocked(state, now_ns, actions);
        state.stable_gesture = label;
        state.stable_since_ns = state.candidate_since_ns;
        state.last_stable_confident_ns = now_ns;
        state.stable_confidence = confidence;
        state.stable_hand = hand ? hand->handedness : PhysicalHandedness::Unknown;
        state.stable_pointer_x = pointer_x;
        state.stable_pointer_y = pointer_y;
        state.stable_has_pointer = has_pointer;
        state.status.stable_gesture = label;
        state.status.active_hand = state.stable_hand;
        PublishAndRouteLocked(state, PhysicalGesturePhase::Started,
                              now_ns, actions);
    }
}

void ApplyStalenessLocked(PhysicalGestureControlState& state,
                          uint64_t now_ns,
                          std::vector<PendingAction>& actions)
{
    if (!state.stable_gesture.empty()) {
        const uint64_t confidence_age = state.last_stable_confident_ns != 0
            ? now_ns - state.last_stable_confident_ns : 0;
        const uint64_t result_age = state.last_result_seen_ns != 0
            ? now_ns - state.last_result_seen_ns : 0;
        if (confidence_age >= MillisecondsToNs(state.config.release_grace_ms) ||
            result_age >= MillisecondsToNs(state.config.result_stale_ms)) {
            ReleaseStableLocked(state, now_ns, actions);
        }
        if (result_age >= MillisecondsToNs(state.config.result_stale_ms)) {
            state.candidate_gesture.clear();
            state.candidate_since_ns = 0;
            state.candidate_hand = PhysicalHandedness::Unknown;
            state.status.candidate_gesture.clear();
        }
    }
    if (state.stable_gesture.empty() && state.last_result_seen_ns != 0 &&
        now_ns - state.last_result_seen_ns >=
            MillisecondsToNs(state.config.result_stale_ms)) {
        state.candidate_gesture.clear();
        state.candidate_since_ns = 0;
        state.candidate_hand = PhysicalHandedness::Unknown;
        state.status.candidate_gesture.clear();
    }
    if (state.status.armed && state.status.armed_until_steady_ns != 0 &&
        now_ns >= state.status.armed_until_steady_ns) {
        DisarmLocked(state, "armed session timed out");
    }
}

bool ExecutePendingAction(const PendingAction& action) {
    switch (action.type) {
        case PendingActionType::MovePointer:
            return PlatformInput::moveCursorRelative(action.dx, action.dy);
        case PendingActionType::LeftClick:
            return PlatformInput::emitMouseClick(0);
        case PendingActionType::RightClick:
            return PlatformInput::emitMouseClick(1);
        case PendingActionType::WakeVoice:
            return WakeKey::requestWake("physical_gesture");
    }
    return false;
}

} // namespace

void TickPhysicalGestureControl() {
    auto& state = State();
    PhysicalHandGestureBus::SnapshotView view;
    bool have_new_snapshot = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        have_new_snapshot = PhysicalHandGestureBus::Instance()
            .PullLatestPhysicalHandGestureSnapshot(
                view, state.last_hand_bus_sequence);
    }

    const uint64_t now_ns = SteadyNowNs();
    std::vector<PendingAction> actions;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.status.enabled = state.config.enabled;
        if (!state.config.enabled) {
            if (state.status.armed) DisarmLocked(state, "controller disabled");
            ReleaseStableLocked(state, now_ns, actions);
            return;
        }

        if (have_new_snapshot &&
            view.snapshot.backend_state == PhysicalHandGestureBackendState::Ready &&
            view.snapshot.source_frame_counter != 0 &&
            view.snapshot.source_frame_counter != state.last_processed_source_frame) {
            state.last_processed_source_frame = view.snapshot.source_frame_counter;
            ProcessHandSnapshotLocked(state, view.snapshot, now_ns, actions);
        }
        ApplyStalenessLocked(state, now_ns, actions);
    }

    for (const auto& action : actions) {
        const bool ok = ExecutePendingAction(action);
        std::lock_guard<std::mutex> lock(state.mutex);
        state.status.last_action = action.description;
        if (ok) {
            ++state.status.actions_executed;
            state.status.last_error.clear();
            state.last_logged_failure_action.clear();
        } else {
            ++state.status.actions_failed;
            state.status.last_error = action.description + " could not be emitted";
            if (state.last_logged_failure_action != action.description) {
                LOG_ERROR(kLogTag, state.status.last_error);
                state.last_logged_failure_action = action.description;
            }
        }
    }
}

PhysicalGestureControlConfig GetPhysicalGestureControlConfig() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.config;
}

PhysicalGestureControlStatus GetPhysicalGestureControlStatus() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.status;
}

void RequestConfigurePhysicalGestureControl(
    const PhysicalGestureControlConfig& requested)
{
    auto config = requested;
    config.enter_confidence = std::clamp(config.enter_confidence, 0.0f, 1.0f);
    config.exit_confidence = std::clamp(config.exit_confidence, 0.0f,
                                        config.enter_confidence);
    config.activation_dwell_ms = std::clamp<uint64_t>(
        config.activation_dwell_ms, 50, 5000);
    config.release_grace_ms = std::clamp<uint64_t>(
        config.release_grace_ms, 50, 5000);
    config.result_stale_ms = std::clamp<uint64_t>(
        config.result_stale_ms, config.release_grace_ms, 10000);
    config.arm_hold_ms = std::clamp<uint64_t>(config.arm_hold_ms, 250, 10000);
    config.disarm_hold_ms = std::clamp<uint64_t>(config.disarm_hold_ms, 250, 10000);
    config.armed_timeout_ms = std::clamp<uint64_t>(
        config.armed_timeout_ms, 1000, 300000);
    config.pointer_gain_pixels = std::clamp(config.pointer_gain_pixels,
                                            100.0f, 10000.0f);
    config.pointer_smoothing = std::clamp(config.pointer_smoothing, 0.0f, 1.0f);
    config.pointer_deadzone_normalized = std::clamp(
        config.pointer_deadzone_normalized, 0.0f, 0.1f);
    config.max_pointer_step_pixels = std::clamp(
        config.max_pointer_step_pixels, 1, 500);
    config.click_cooldown_ms = std::clamp<uint64_t>(
        config.click_cooldown_ms, 100, 10000);
    config.wake_hold_ms = std::clamp<uint64_t>(config.wake_hold_ms, 250, 10000);
    config.wake_cooldown_ms = std::clamp<uint64_t>(
        config.wake_cooldown_ms, 1000, 300000);

    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = std::move(config);
    DisarmLocked(state, "controller configuration changed");
    state.candidate_gesture.clear();
    state.stable_gesture.clear();
    state.candidate_since_ns = 0;
    state.stable_since_ns = 0;
    state.last_stable_confident_ns = 0;
    state.last_processed_source_frame = 0;
    state.status.candidate_gesture.clear();
    state.status.stable_gesture.clear();
    state.status.enabled = state.config.enabled;
}

void RequestSetPhysicalGestureControlEnabled(bool enabled) {
    auto config = GetPhysicalGestureControlConfig();
    config.enabled = enabled;
    RequestConfigurePhysicalGestureControl(config);
}

void ShutdownPhysicalGestureControl() {
    auto& state = State();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        DisarmLocked(state, "controller shutdown");
        state.last_hand_bus_sequence = 0;
        state.last_processed_source_frame = 0;
        state.last_result_seen_ns = 0;
        state.candidate_gesture.clear();
        state.stable_gesture.clear();
        state.status.candidate_gesture.clear();
        state.status.stable_gesture.clear();
        state.status.active_hand = PhysicalHandedness::Unknown;
        state.last_logged_failure_action.clear();
    }
    PhysicalGestureEventBus::Instance().ResetPhysicalGestureEventBus();
}

}}} // namespace GRIM::Perception::Physical
