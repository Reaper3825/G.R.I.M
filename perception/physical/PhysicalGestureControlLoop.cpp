#include "PhysicalGestureControlLoop.hpp"

#include "PhysicalGestureEventBus.hpp"
#include "PhysicalHandGestureBus.hpp"
#include "core/platform_input.hpp"
#include "logger.hpp"
#include "wake/wake_key.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

std::string SanitizeBindingId(std::string value) {
    for (char& ch : value) {
        const auto uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_' && ch != '-' && ch != '.') ch = '_';
    }
    return value;
}

struct PendingAction {
    PhysicalGestureAction type = PhysicalGestureAction::PointerMove;
    int dx = 0;
    int dy = 0;
    std::string description;
    std::string binding_id;
};

float OneEuroAlpha(float cutoff_hz, float dt_seconds) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float safe_cutoff = std::max(0.001f, cutoff_hz);
    const float tau = 1.0f / (kTwoPi * safe_cutoff);
    return 1.0f / (1.0f + tau / std::max(0.0001f, dt_seconds));
}

struct OneEuroAxis {
    bool initialized = false;
    float raw = 0.0f;
    float filtered = 0.0f;
    float derivative = 0.0f;
    uint64_t sample_ns = 0;

    void Reset() {
        initialized = false;
        raw = 0.0f;
        filtered = 0.0f;
        derivative = 0.0f;
        sample_ns = 0;
    }

    float Filter(float value, uint64_t now_ns,
                 float min_cutoff, float beta, float derivative_cutoff) {
        if (!initialized || now_ns <= sample_ns) {
            initialized = true;
            raw = value;
            filtered = value;
            derivative = 0.0f;
            sample_ns = now_ns;
            return filtered;
        }
        const float dt = std::clamp(
            static_cast<float>(now_ns - sample_ns) / 1.0e9f, 0.001f, 0.25f);
        const float raw_derivative = (value - raw) / dt;
        const float derivative_alpha = OneEuroAlpha(derivative_cutoff, dt);
        derivative += derivative_alpha * (raw_derivative - derivative);
        const float cutoff = min_cutoff + beta * std::abs(derivative);
        const float value_alpha = OneEuroAlpha(cutoff, dt);
        filtered += value_alpha * (value - filtered);
        raw = value;
        sample_ns = now_ns;
        return filtered;
    }
};

struct PhysicalGestureControlState {
    std::mutex mutex;
    PhysicalGestureControlConfig config = DefaultPhysicalGestureControlConfig();
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

    std::unordered_set<std::string> latched_bindings;
    std::unordered_map<std::string, uint64_t> last_binding_action_ns;

    OneEuroAxis pointer_filter_x;
    OneEuroAxis pointer_filter_y;
    bool pointer_motion_active = false;
    uint64_t last_pointer_hand_seen_ns = 0;
    uint64_t last_pointer_observation_ns = 0;
    std::string active_pointer_binding_id;
    std::string active_pointer_gesture;

    bool hand_lock_active = false;
    bool ever_had_hand_lock = false;
    PhysicalHandedness locked_handedness = PhysicalHandedness::Unknown;
    float locked_wrist_x = 0.0f;
    float locked_wrist_y = 0.0f;
    uint64_t last_locked_hand_seen_ns = 0;
    std::string last_logged_failure_action;
};

PhysicalGestureControlState& State() {
    static PhysicalGestureControlState state;
    return state;
}

void ResetPointerLocked(PhysicalGestureControlState& state) {
    state.status.pointer_active = false;
    state.status.pointer_activation_gesture.clear();
    state.pointer_filter_x.Reset();
    state.pointer_filter_y.Reset();
    state.pointer_motion_active = false;
    state.last_pointer_hand_seen_ns = 0;
    state.last_pointer_observation_ns = 0;
    state.active_pointer_binding_id.clear();
    state.active_pointer_gesture.clear();
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

bool BindingHandMatches(const PhysicalGestureBinding& binding,
                        PhysicalHandedness handedness)
{
    return binding.handedness == PhysicalHandedness::Unknown ||
           binding.handedness == handedness;
}

bool BindingTriggerMatches(const PhysicalGestureBinding& binding,
                           const PhysicalGestureEvent& event)
{
    switch (binding.trigger) {
        case PhysicalGestureTrigger::Started:
            return event.phase == PhysicalGesturePhase::Started;
        case PhysicalGestureTrigger::Held:
            return event.phase != PhysicalGesturePhase::Released &&
                   event.held_ms >= binding.minimum_hold_ms;
        case PhysicalGestureTrigger::Released:
            return event.phase == PhysicalGesturePhase::Released &&
                   event.held_ms >= binding.minimum_hold_ms;
    }
    return false;
}

void RecordInternalActionLocked(PhysicalGestureControlState& state,
                                const PhysicalGestureBinding& binding)
{
    const std::string action_id = PhysicalGestureActionId(binding.action);
    if (state.config.dry_run) {
        ++state.status.actions_previewed;
        state.status.last_action = "dry_run:" + action_id;
    } else {
        ++state.status.actions_executed;
        state.status.last_action = action_id;
    }
    state.status.last_block_reason.clear();
}

bool BindingCoolingDownLocked(const PhysicalGestureControlState& state,
                              const PhysicalGestureBinding& binding,
                              uint64_t now_ns)
{
    const auto it = state.last_binding_action_ns.find(binding.binding_id);
    return binding.cooldown_ms != 0 &&
           it != state.last_binding_action_ns.end() &&
           now_ns - it->second < MillisecondsToNs(binding.cooldown_ms);
}

bool ControlPointForHand(const PhysicalHandObservation& hand,
                         const PhysicalGestureControlConfig& config,
                         float& x, float& y)
{
    if (hand.landmark_count <= 8) return false;
    const float tip_weight = std::clamp(config.pointer_tip_weight, 0.0f, 1.0f);
    const auto& tip = hand.landmarks[8];
    const auto& pip = hand.landmarks[6];
    x = tip_weight * tip.normalized_x +
        (1.0f - tip_weight) * pip.normalized_x;
    y = tip_weight * tip.normalized_y +
        (1.0f - tip_weight) * pip.normalized_y;
    return std::isfinite(x) && std::isfinite(y);
}

void LockHandLocked(PhysicalGestureControlState& state,
                    const PhysicalHandObservation& hand,
                    uint64_t now_ns)
{
    if (!state.hand_lock_active && state.ever_had_hand_lock)
        ++state.status.hand_reacquisitions;
    state.hand_lock_active = true;
    state.ever_had_hand_lock = true;
    state.locked_handedness = hand.handedness;
    if (hand.landmark_count > 0) {
        state.locked_wrist_x = hand.landmarks[0].normalized_x;
        state.locked_wrist_y = hand.landmarks[0].normalized_y;
    }
    state.last_locked_hand_seen_ns = now_ns;
    state.status.hand_lock_active = true;
    state.status.active_hand = hand.handedness;
}

void ReleaseHandLockLocked(PhysicalGestureControlState& state) {
    state.hand_lock_active = false;
    state.locked_handedness = PhysicalHandedness::Unknown;
    state.last_locked_hand_seen_ns = 0;
    state.status.hand_lock_active = false;
    if (state.stable_gesture.empty())
        state.status.active_hand = PhysicalHandedness::Unknown;
}

void BeginPointerLocked(PhysicalGestureControlState& state,
                        const PhysicalGestureBinding& binding,
                        const PhysicalGestureEvent& event)
{
    state.pointer_filter_x.Reset();
    state.pointer_filter_y.Reset();
    state.pointer_filter_x.Filter(
        event.pointer_x, event.event_steady_ns,
        state.config.pointer_one_euro_min_cutoff,
        state.config.pointer_one_euro_beta,
        state.config.pointer_one_euro_derivative_cutoff);
    state.pointer_filter_y.Filter(
        event.pointer_y, event.event_steady_ns,
        state.config.pointer_one_euro_min_cutoff,
        state.config.pointer_one_euro_beta,
        state.config.pointer_one_euro_derivative_cutoff);
    state.pointer_motion_active = false;
    state.last_pointer_hand_seen_ns = event.event_steady_ns;
    state.last_pointer_observation_ns = event.event_steady_ns;
    state.active_pointer_binding_id = binding.binding_id;
    state.active_pointer_gesture = event.gesture_label;
    state.status.pointer_active = true;
    state.status.pointer_activation_gesture = event.gesture_label;
    state.status.pointer_raw_x = event.pointer_x;
    state.status.pointer_raw_y = event.pointer_y;
    state.status.pointer_filtered_x = event.pointer_x;
    state.status.pointer_filtered_y = event.pointer_y;
    state.status.pointer_sample_hz = 0.0f;
    ++state.status.pointer_samples;
}

void ProcessPointerSampleLocked(PhysicalGestureControlState& state,
                                float raw_x, float raw_y,
                                uint64_t sample_ns,
                                bool classifier_supports_pointer,
                                std::vector<PendingAction>& actions)
{
    if (!state.status.pointer_active) return;
    state.last_pointer_hand_seen_ns = sample_ns;
    ++state.status.pointer_samples;
    if (!classifier_supports_pointer)
        ++state.status.pointer_classifier_bypass_frames;

    if (state.last_pointer_observation_ns != 0 &&
        sample_ns > state.last_pointer_observation_ns) {
        const float observation_dt = std::max(0.001f,
            static_cast<float>(sample_ns - state.last_pointer_observation_ns) / 1.0e9f);
        const float instant_hz = 1.0f / observation_dt;
        state.status.pointer_sample_hz = state.status.pointer_sample_hz <= 0.0f
            ? instant_hz
            : 0.85f * state.status.pointer_sample_hz + 0.15f * instant_hz;
    }
    state.last_pointer_observation_ns = sample_ns;

    const bool initialized = state.pointer_filter_x.initialized &&
                             state.pointer_filter_y.initialized;
    if (initialized && sample_ns > state.pointer_filter_x.sample_ns) {
        const float dt = std::max(0.001f,
            static_cast<float>(sample_ns - state.pointer_filter_x.sample_ns) / 1.0e9f);
        const float distance = std::hypot(
            raw_x - state.pointer_filter_x.raw,
            raw_y - state.pointer_filter_y.raw);
        const float velocity = distance / dt;
        if (velocity > state.config.pointer_outlier_velocity_normalized_per_second) {
            ++state.status.pointer_outliers_rejected;
            return;
        }
    }

    const float previous_x = state.pointer_filter_x.filtered;
    const float previous_y = state.pointer_filter_y.filtered;
    const float filtered_x = state.pointer_filter_x.Filter(
        raw_x, sample_ns,
        state.config.pointer_one_euro_min_cutoff,
        state.config.pointer_one_euro_beta,
        state.config.pointer_one_euro_derivative_cutoff);
    const float filtered_y = state.pointer_filter_y.Filter(
        raw_y, sample_ns,
        state.config.pointer_one_euro_min_cutoff,
        state.config.pointer_one_euro_beta,
        state.config.pointer_one_euro_derivative_cutoff);
    state.status.pointer_raw_x = raw_x;
    state.status.pointer_raw_y = raw_y;
    state.status.pointer_filtered_x = filtered_x;
    state.status.pointer_filtered_y = filtered_y;
    if (!initialized) return;

    float nx = filtered_x - previous_x;
    float ny = filtered_y - previous_y;
    const float magnitude = std::hypot(nx, ny);
    const float base_deadzone = state.config.pointer_deadzone_normalized;
    const float start_deadzone = base_deadzone *
        state.config.pointer_deadzone_start_multiplier;
    if (!state.pointer_motion_active) {
        if (magnitude < start_deadzone) return;
        state.pointer_motion_active = true;
    } else if (magnitude < base_deadzone) {
        state.pointer_motion_active = false;
        return;
    }

    if (state.config.invert_pointer_x) nx = -nx;
    if (state.config.invert_pointer_y) ny = -ny;
    const int limit = std::max(1, state.config.max_pointer_step_pixels);
    const int dx = std::clamp(static_cast<int>(std::lround(
        nx * state.config.pointer_gain_pixels)), -limit, limit);
    const int dy = std::clamp(static_cast<int>(std::lround(
        ny * state.config.pointer_gain_pixels)), -limit, limit);
    if (dx == 0 && dy == 0) return;
    ++state.status.pointer_moves_emitted;
    actions.push_back({PhysicalGestureAction::PointerMove, dx, dy,
        PhysicalGestureActionId(PhysicalGestureAction::PointerMove),
        state.active_pointer_binding_id});
    ExtendArmedWindowLocked(state, sample_ns);
}

void RouteEventLocked(PhysicalGestureControlState& state,
                      const PhysicalGestureEvent& event,
                      std::vector<PendingAction>& actions)
{
    const auto& config = state.config;
    const uint64_t now_ns = event.event_steady_ns;

    if (event.phase == PhysicalGesturePhase::Released) {
        for (const auto& binding : config.bindings) {
            if (binding.gesture_label != event.gesture_label ||
                !BindingHandMatches(binding, event.handedness)) continue;
            state.latched_bindings.erase(binding.binding_id);
        }
    }

    for (const auto& binding : config.bindings) {
        if (!binding.enabled || binding.gesture_label != event.gesture_label ||
            !BindingHandMatches(binding, event.handedness)) continue;

        if (binding.action == PhysicalGestureAction::PointerMove) {
            if (event.phase == PhysicalGesturePhase::Released) continue;
            if (event.held_ms < binding.minimum_hold_ms) break;
            if (binding.requires_armed && !state.status.armed) {
                if (event.phase == PhysicalGesturePhase::Started)
                    RecordBlockedLocked(state, binding.binding_id +
                        ": control is not armed");
                break;
            }
            if (!event.has_pointer) break;
            ExtendArmedWindowLocked(state, now_ns);
            if (!state.status.pointer_active)
                BeginPointerLocked(state, binding, event);
            break;
        }

        if (!BindingTriggerMatches(binding, event)) continue;
        if (state.latched_bindings.count(binding.binding_id) != 0) break;
        state.latched_bindings.insert(binding.binding_id);

        if (binding.requires_armed && !state.status.armed) {
            RecordBlockedLocked(state, binding.binding_id +
                ": action requires armed control");
            break;
        }
        if (BindingCoolingDownLocked(state, binding, now_ns)) {
            RecordBlockedLocked(state, binding.binding_id +
                ": action is cooling down");
            break;
        }

        state.last_binding_action_ns[binding.binding_id] = now_ns;
        switch (binding.action) {
            case PhysicalGestureAction::ControlArm:
                state.status.armed = true;
                ExtendArmedWindowLocked(state, now_ns);
                RecordInternalActionLocked(state, binding);
                LOG_DEBUG(kLogTag, "Gesture control armed by " + binding.binding_id);
                break;
            case PhysicalGestureAction::ControlDisarm:
                RecordInternalActionLocked(state, binding);
                DisarmLocked(state, "explicit disarm binding " + binding.binding_id);
                break;
            case PhysicalGestureAction::MouseLeftClick:
            case PhysicalGestureAction::MouseRightClick:
                ExtendArmedWindowLocked(state, now_ns);
                ResetPointerLocked(state);
                actions.push_back({binding.action, 0, 0,
                    PhysicalGestureActionId(binding.action), binding.binding_id});
                break;
            case PhysicalGestureAction::VoiceWake:
                actions.push_back({binding.action, 0, 0,
                    PhysicalGestureActionId(binding.action), binding.binding_id});
                break;
            case PhysicalGestureAction::PointerMove:
                break;
        }
        break;
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
    state.status.active_hand = state.hand_lock_active
        ? state.locked_handedness : PhysicalHandedness::Unknown;
}

const PhysicalHandObservation* SelectControlHandLocked(
    PhysicalGestureControlState& state,
    const PhysicalHandGestureSnapshot& snapshot,
    uint64_t now_ns)
{
    const auto& config = state.config;
    if (state.hand_lock_active) {
        const PhysicalHandObservation* nearest = nullptr;
        float nearest_distance = 0.0f;
        for (const auto& hand : snapshot.hands) {
            if (config.preferred_hand != PhysicalHandedness::Unknown &&
                hand.handedness != config.preferred_hand) continue;
            if (hand.landmark_count == 0) continue;
            const float dx = hand.landmarks[0].normalized_x - state.locked_wrist_x;
            const float dy = hand.landmarks[0].normalized_y - state.locked_wrist_y;
            float distance = std::hypot(dx, dy);
            if (state.locked_handedness != PhysicalHandedness::Unknown &&
                hand.handedness != PhysicalHandedness::Unknown &&
                hand.handedness != state.locked_handedness) {
                distance += 0.08f;
            }
            if (!nearest || distance < nearest_distance) {
                nearest = &hand;
                nearest_distance = distance;
            }
        }
        if (nearest && nearest_distance <=
            config.pointer_hand_lock_radius_normalized) {
            state.locked_wrist_x = nearest->landmarks[0].normalized_x;
            state.locked_wrist_y = nearest->landmarks[0].normalized_y;
            state.last_locked_hand_seen_ns = now_ns;
            return nearest;
        }
        return nullptr;
    }

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
    const auto* hand = SelectControlHandLocked(state, snapshot, now_ns);
    const std::string label = hand
        ? UsableGestureLabel(hand->gesture_label) : std::string{};
    const float confidence = hand ? hand->gesture_confidence : 0.0f;
    float pointer_x = 0.0f;
    float pointer_y = 0.0f;
    const bool has_pointer = hand &&
        ControlPointForHand(*hand, state.config, pointer_x, pointer_y);

    if (state.hand_lock_active && !hand &&
        state.last_locked_hand_seen_ns != 0 &&
        now_ns - state.last_locked_hand_seen_ns >=
            MillisecondsToNs(state.config.pointer_hand_loss_grace_ms)) {
        ResetPointerLocked(state);
        ReleaseHandLockLocked(state);
    }

    if (state.status.pointer_active && hand && has_pointer) {
        const bool classifier_supports_pointer =
            label == state.active_pointer_gesture &&
            confidence >= state.config.exit_confidence;
        ProcessPointerSampleLocked(state, pointer_x, pointer_y, now_ns,
                                   classifier_supports_pointer, actions);
    }

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
        if (state.status.pointer_active &&
            label != state.active_pointer_gesture) {
            ResetPointerLocked(state);
        }
        ReleaseStableLocked(state, now_ns, actions);
        state.stable_gesture = label;
        state.stable_since_ns = state.candidate_since_ns;
        state.last_stable_confident_ns = now_ns;
        state.stable_confidence = confidence;
        state.stable_hand = hand ? hand->handedness : PhysicalHandedness::Unknown;
        state.stable_pointer_x = pointer_x;
        state.stable_pointer_y = pointer_y;
        state.stable_has_pointer = has_pointer;
        if (hand) LockHandLocked(state, *hand, now_ns);
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
    if (state.status.pointer_active && state.last_pointer_hand_seen_ns != 0 &&
        now_ns - state.last_pointer_hand_seen_ns >=
            MillisecondsToNs(state.config.pointer_hand_loss_grace_ms)) {
        ResetPointerLocked(state);
    }
    if (state.hand_lock_active && state.last_locked_hand_seen_ns != 0 &&
        now_ns - state.last_locked_hand_seen_ns >=
            MillisecondsToNs(state.config.pointer_hand_loss_grace_ms)) {
        ReleaseHandLockLocked(state);
    }
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
        case PhysicalGestureAction::PointerMove:
            return PlatformInput::moveCursorRelative(action.dx, action.dy);
        case PhysicalGestureAction::MouseLeftClick:
            return PlatformInput::emitMouseClick(0);
        case PhysicalGestureAction::MouseRightClick:
            return PlatformInput::emitMouseClick(1);
        case PhysicalGestureAction::VoiceWake:
            return WakeKey::requestWake(action.binding_id.empty()
                ? "physical_gesture"
                : "physical_gesture:" + action.binding_id);
        case PhysicalGestureAction::ControlArm:
        case PhysicalGestureAction::ControlDisarm:
            return true;
    }
    return false;
}

} // namespace

std::vector<PhysicalGestureBinding> DefaultPhysicalGestureBindings() {
    return {
        {"arm", "Victory", PhysicalGestureAction::ControlArm,
         PhysicalGestureTrigger::Held, PhysicalHandedness::Unknown,
         800, 1000, false, true, 100},
        {"disarm", "Open_Palm", PhysicalGestureAction::ControlDisarm,
         PhysicalGestureTrigger::Held, PhysicalHandedness::Unknown,
         700, 500, false, true, 110},
        {"pointer", "Pointing_Up", PhysicalGestureAction::PointerMove,
         PhysicalGestureTrigger::Held, PhysicalHandedness::Unknown,
         0, 0, true, true, 50},
        {"left_click", "Closed_Fist", PhysicalGestureAction::MouseLeftClick,
         PhysicalGestureTrigger::Started, PhysicalHandedness::Unknown,
         0, 650, true, true, 60},
        {"right_click", "Thumb_Down", PhysicalGestureAction::MouseRightClick,
         PhysicalGestureTrigger::Started, PhysicalHandedness::Unknown,
         0, 650, true, true, 60},
        {"wake", "ILoveYou", PhysicalGestureAction::VoiceWake,
         PhysicalGestureTrigger::Held, PhysicalHandedness::Unknown,
         1100, 10000, false, true, 80}
    };
}

PhysicalGestureControlConfig DefaultPhysicalGestureControlConfig() {
    PhysicalGestureControlConfig config;
    config.bindings = DefaultPhysicalGestureBindings();
    return config;
}

const char* PhysicalGestureActionId(PhysicalGestureAction action) noexcept {
    switch (action) {
        case PhysicalGestureAction::ControlArm: return "control.arm";
        case PhysicalGestureAction::ControlDisarm: return "control.disarm";
        case PhysicalGestureAction::PointerMove: return "pointer.move";
        case PhysicalGestureAction::MouseLeftClick: return "mouse.left_click";
        case PhysicalGestureAction::MouseRightClick: return "mouse.right_click";
        case PhysicalGestureAction::VoiceWake: return "voice.wake";
    }
    return "unknown";
}

const char* PhysicalGestureActionDisplayName(PhysicalGestureAction action) noexcept {
    switch (action) {
        case PhysicalGestureAction::ControlArm: return "Arm control";
        case PhysicalGestureAction::ControlDisarm: return "Disarm control";
        case PhysicalGestureAction::PointerMove: return "Move pointer";
        case PhysicalGestureAction::MouseLeftClick: return "Left click";
        case PhysicalGestureAction::MouseRightClick: return "Right click";
        case PhysicalGestureAction::VoiceWake: return "Wake voice";
    }
    return "Unknown";
}

bool TryParsePhysicalGestureAction(const std::string& value,
                                   PhysicalGestureAction& action) noexcept
{
    for (int i = static_cast<int>(PhysicalGestureAction::ControlArm);
         i <= static_cast<int>(PhysicalGestureAction::VoiceWake); ++i) {
        const auto candidate = static_cast<PhysicalGestureAction>(i);
        if (value == PhysicalGestureActionId(candidate)) {
            action = candidate;
            return true;
        }
    }
    return false;
}

const char* PhysicalGestureTriggerId(PhysicalGestureTrigger trigger) noexcept {
    switch (trigger) {
        case PhysicalGestureTrigger::Started: return "started";
        case PhysicalGestureTrigger::Held: return "held";
        case PhysicalGestureTrigger::Released: return "released";
    }
    return "unknown";
}

bool TryParsePhysicalGestureTrigger(const std::string& value,
                                    PhysicalGestureTrigger& trigger) noexcept
{
    if (value == "started") {
        trigger = PhysicalGestureTrigger::Started;
        return true;
    }
    if (value == "held") {
        trigger = PhysicalGestureTrigger::Held;
        return true;
    }
    if (value == "released") {
        trigger = PhysicalGestureTrigger::Released;
        return true;
    }
    return false;
}

std::vector<std::string> ValidatePhysicalGestureBindings(
    const PhysicalGestureControlConfig& config)
{
    std::vector<std::string> issues;
    std::unordered_set<std::string> ids;
    for (size_t i = 0; i < config.bindings.size(); ++i) {
        const auto& binding = config.bindings[i];
        if (binding.binding_id.empty())
            issues.push_back("Binding " + std::to_string(i + 1) + " has no id");
        else if (!ids.insert(binding.binding_id).second)
            issues.push_back("Duplicate binding id: " + binding.binding_id);
        if (binding.enabled && binding.gesture_label.empty())
            issues.push_back("Enabled binding " + binding.binding_id +
                             " has no gesture label");

        for (size_t j = i + 1; binding.enabled && j < config.bindings.size(); ++j) {
            const auto& other = config.bindings[j];
            if (!other.enabled || binding.gesture_label.empty() ||
                binding.gesture_label != other.gesture_label) continue;
            const bool hands_overlap =
                binding.handedness == PhysicalHandedness::Unknown ||
                other.handedness == PhysicalHandedness::Unknown ||
                binding.handedness == other.handedness;
            const bool phases_overlap = binding.trigger == other.trigger ||
                binding.action == PhysicalGestureAction::PointerMove ||
                other.action == PhysicalGestureAction::PointerMove;
            if (hands_overlap && phases_overlap) {
                issues.push_back("Conflict: " + binding.binding_id + " and " +
                    other.binding_id + " overlap for the same gesture");
            }
        }
    }
    return issues;
}

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
        state.status.dry_run = state.config.dry_run;
        if (!state.config.enabled) {
            if (state.status.armed) DisarmLocked(state, "controller disabled");
            ReleaseStableLocked(state, now_ns, actions);
            ReleaseHandLockLocked(state);
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
        bool dry_run = false;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            dry_run = state.config.dry_run;
        }
        const bool ok = dry_run || ExecutePendingAction(action);
        std::lock_guard<std::mutex> lock(state.mutex);
        state.status.last_action = dry_run
            ? "dry_run:" + action.description : action.description;
        if (ok) {
            if (dry_run) ++state.status.actions_previewed;
            else ++state.status.actions_executed;
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
    config.armed_timeout_ms = std::clamp<uint64_t>(
        config.armed_timeout_ms, 1000, 300000);
    config.pointer_gain_pixels = std::clamp(config.pointer_gain_pixels,
                                            100.0f, 10000.0f);
    config.pointer_smoothing = std::clamp(config.pointer_smoothing, 0.0f, 1.0f);
    config.pointer_deadzone_normalized = std::clamp(
        config.pointer_deadzone_normalized, 0.0f, 0.1f);
    config.pointer_deadzone_start_multiplier = std::clamp(
        config.pointer_deadzone_start_multiplier, 1.0f, 5.0f);
    config.pointer_one_euro_min_cutoff = std::clamp(
        config.pointer_one_euro_min_cutoff, 0.05f, 20.0f);
    config.pointer_one_euro_beta = std::clamp(
        config.pointer_one_euro_beta, 0.0f, 2.0f);
    config.pointer_one_euro_derivative_cutoff = std::clamp(
        config.pointer_one_euro_derivative_cutoff, 0.05f, 20.0f);
    config.pointer_outlier_velocity_normalized_per_second = std::clamp(
        config.pointer_outlier_velocity_normalized_per_second, 0.25f, 20.0f);
    config.pointer_tip_weight = std::clamp(config.pointer_tip_weight, 0.0f, 1.0f);
    config.pointer_hand_lock_radius_normalized = std::clamp(
        config.pointer_hand_lock_radius_normalized, 0.05f, 1.0f);
    config.pointer_hand_loss_grace_ms = std::clamp<uint64_t>(
        config.pointer_hand_loss_grace_ms, 100, 3000);
    config.max_pointer_step_pixels = std::clamp(
        config.max_pointer_step_pixels, 1, 500);
    if (config.bindings.empty()) config.bindings = DefaultPhysicalGestureBindings();
    std::unordered_set<std::string> used_ids;
    for (size_t i = 0; i < config.bindings.size(); ++i) {
        auto& binding = config.bindings[i];
        binding.binding_id = SanitizeBindingId(std::move(binding.binding_id));
        if (binding.binding_id.empty())
            binding.binding_id = "binding_" + std::to_string(i + 1);
        const std::string base_id = binding.binding_id;
        int suffix = 2;
        while (!used_ids.insert(binding.binding_id).second)
            binding.binding_id = base_id + "_" + std::to_string(suffix++);
        binding.minimum_hold_ms = std::clamp<uint64_t>(
            binding.minimum_hold_ms, 0, 10000);
        binding.cooldown_ms = std::clamp<uint64_t>(
            binding.cooldown_ms, 0, 300000);
        binding.priority = std::clamp(binding.priority, -1000, 1000);
        if (binding.trigger == PhysicalGestureTrigger::Started)
            binding.minimum_hold_ms = 0;
        if (binding.action == PhysicalGestureAction::PointerMove) {
            binding.trigger = PhysicalGestureTrigger::Held;
            binding.requires_armed = true;
        }
        if (binding.action == PhysicalGestureAction::MouseLeftClick ||
            binding.action == PhysicalGestureAction::MouseRightClick) {
            binding.requires_armed = true;
        }
        if (binding.action == PhysicalGestureAction::ControlArm ||
            binding.action == PhysicalGestureAction::ControlDisarm) {
            binding.requires_armed = false;
        }
    }
    std::stable_sort(config.bindings.begin(), config.bindings.end(),
        [](const PhysicalGestureBinding& a, const PhysicalGestureBinding& b) {
            return a.priority > b.priority;
        });

    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.config = std::move(config);
    state.latched_bindings.clear();
    state.last_binding_action_ns.clear();
    DisarmLocked(state, "controller configuration changed");
    ReleaseHandLockLocked(state);
    state.candidate_gesture.clear();
    state.stable_gesture.clear();
    state.candidate_since_ns = 0;
    state.stable_since_ns = 0;
    state.last_stable_confident_ns = 0;
    state.last_processed_source_frame = 0;
    state.ever_had_hand_lock = false;
    state.status.candidate_gesture.clear();
    state.status.stable_gesture.clear();
    state.status.active_hand = PhysicalHandedness::Unknown;
    state.status.enabled = state.config.enabled;
    state.status.dry_run = state.config.dry_run;
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
        ReleaseHandLockLocked(state);
        state.ever_had_hand_lock = false;
        state.latched_bindings.clear();
        state.last_binding_action_ns.clear();
        state.last_logged_failure_action.clear();
    }
    PhysicalGestureEventBus::Instance().ResetPhysicalGestureEventBus();
}

}}} // namespace GRIM::Perception::Physical
