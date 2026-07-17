#include "PhysicalGestureControlConfigIO.hpp"

#include "PhysicalGestureControlLoop.hpp"
#include "settings/runtime_ai_config.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

const char* HandednessId(PhysicalHandedness handedness) {
    switch (handedness) {
        case PhysicalHandedness::Unknown: return "either";
        case PhysicalHandedness::Left: return "left";
        case PhysicalHandedness::Right: return "right";
    }
    return "either";
}

PhysicalHandedness ParseHandedness(const std::string& value) {
    if (value == "either" || value == "unknown")
        return PhysicalHandedness::Unknown;
    if (value == "left") return PhysicalHandedness::Left;
    if (value == "right") return PhysicalHandedness::Right;
    throw std::runtime_error("unknown gesture handedness: " + value);
}

nlohmann::json BindingToJson(const PhysicalGestureBinding& binding) {
    return {
        {"id", binding.binding_id},
        {"gesture", binding.gesture_label},
        {"action", PhysicalGestureActionId(binding.action)},
        {"trigger", PhysicalGestureTriggerId(binding.trigger)},
        {"hand", HandednessId(binding.handedness)},
        {"minimum_hold_ms", binding.minimum_hold_ms},
        {"cooldown_ms", binding.cooldown_ms},
        {"requires_armed", binding.requires_armed},
        {"enabled", binding.enabled},
        {"priority", binding.priority}
    };
}

PhysicalGestureBinding BindingFromJson(const nlohmann::json& value,
                                       size_t index)
{
    if (!value.is_object())
        throw std::runtime_error("gesture binding must be an object");

    PhysicalGestureBinding binding;
    binding.binding_id = value.value("id", "binding_" + std::to_string(index + 1));
    binding.gesture_label = value.value("gesture", std::string{});
    binding.handedness = ParseHandedness(value.value("hand", "either"));
    binding.minimum_hold_ms = value.value("minimum_hold_ms", uint64_t{0});
    binding.cooldown_ms = value.value("cooldown_ms", uint64_t{0});
    binding.requires_armed = value.value("requires_armed", false);
    binding.enabled = value.value("enabled", true);
    binding.priority = value.value("priority", 0);

    const std::string action_id = value.value("action", "voice.wake");
    if (!TryParsePhysicalGestureAction(action_id, binding.action))
        throw std::runtime_error("unknown gesture action: " + action_id);
    const std::string trigger_id = value.value("trigger", "held");
    if (!TryParsePhysicalGestureTrigger(trigger_id, binding.trigger))
        throw std::runtime_error("unknown gesture trigger: " + trigger_id);
    return binding;
}

} // namespace

bool ApplyPhysicalGestureControlConfigFromRuntime(
    const nlohmann::json& runtime_config, std::string& error)
{
    error.clear();
    try {
        if (!runtime_config.contains("physical_interaction") ||
            !runtime_config.at("physical_interaction").is_object()) return true;
        const auto& physical = runtime_config.at("physical_interaction");
        if (!physical.contains("gesture_control")) return true;
        const auto& value = physical.at("gesture_control");
        if (!value.is_object())
            throw std::runtime_error("physical_interaction.gesture_control must be an object");
        if (value.value("schema", 1) != 1)
            throw std::runtime_error("unsupported gesture-control config schema");

        auto config = DefaultPhysicalGestureControlConfig();
        config.enabled = value.value("enabled", config.enabled);
        config.dry_run = value.value("dry_run", config.dry_run);
        config.preferred_hand = ParseHandedness(
            value.value("preferred_hand", "either"));
        config.enter_confidence = value.value(
            "enter_confidence", config.enter_confidence);
        config.exit_confidence = value.value(
            "exit_confidence", config.exit_confidence);
        config.activation_dwell_ms = value.value(
            "activation_dwell_ms", config.activation_dwell_ms);
        config.release_grace_ms = value.value(
            "release_grace_ms", config.release_grace_ms);
        config.result_stale_ms = value.value(
            "result_stale_ms", config.result_stale_ms);
        config.armed_timeout_ms = value.value(
            "armed_timeout_ms", config.armed_timeout_ms);
        config.pointer_gain_pixels = value.value(
            "pointer_gain_pixels", config.pointer_gain_pixels);
        config.pointer_smoothing = value.value(
            "pointer_smoothing", config.pointer_smoothing);
        config.pointer_deadzone_normalized = value.value(
            "pointer_deadzone_normalized", config.pointer_deadzone_normalized);
        config.pointer_deadzone_start_multiplier = value.value(
            "pointer_deadzone_start_multiplier",
            config.pointer_deadzone_start_multiplier);
        config.pointer_one_euro_min_cutoff = value.value(
            "pointer_one_euro_min_cutoff", config.pointer_one_euro_min_cutoff);
        config.pointer_one_euro_beta = value.value(
            "pointer_one_euro_beta", config.pointer_one_euro_beta);
        config.pointer_one_euro_derivative_cutoff = value.value(
            "pointer_one_euro_derivative_cutoff",
            config.pointer_one_euro_derivative_cutoff);
        config.pointer_outlier_velocity_normalized_per_second = value.value(
            "pointer_outlier_velocity_normalized_per_second",
            config.pointer_outlier_velocity_normalized_per_second);
        config.pointer_tip_weight = value.value(
            "pointer_tip_weight", config.pointer_tip_weight);
        config.pointer_hand_lock_radius_normalized = value.value(
            "pointer_hand_lock_radius_normalized",
            config.pointer_hand_lock_radius_normalized);
        config.pointer_hand_loss_grace_ms = value.value(
            "pointer_hand_loss_grace_ms", config.pointer_hand_loss_grace_ms);
        config.max_pointer_step_pixels = value.value(
            "max_pointer_step_pixels", config.max_pointer_step_pixels);
        config.invert_pointer_x = value.value(
            "invert_pointer_x", config.invert_pointer_x);
        config.invert_pointer_y = value.value(
            "invert_pointer_y", config.invert_pointer_y);

        if (value.contains("bindings")) {
            if (!value.at("bindings").is_array())
                throw std::runtime_error("gesture_control.bindings must be an array");
            config.bindings.clear();
            size_t index = 0;
            for (const auto& item : value.at("bindings"))
                config.bindings.push_back(BindingFromJson(item, index++));
        }
        RequestConfigurePhysicalGestureControl(config);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

nlohmann::json SerializePhysicalGestureControlConfig() {
    const auto config = GetPhysicalGestureControlConfig();
    nlohmann::json bindings = nlohmann::json::array();
    for (const auto& binding : config.bindings)
        bindings.push_back(BindingToJson(binding));

    return {
        {"schema", 1},
        {"enabled", config.enabled},
        {"dry_run", config.dry_run},
        {"preferred_hand", HandednessId(config.preferred_hand)},
        {"enter_confidence", config.enter_confidence},
        {"exit_confidence", config.exit_confidence},
        {"activation_dwell_ms", config.activation_dwell_ms},
        {"release_grace_ms", config.release_grace_ms},
        {"result_stale_ms", config.result_stale_ms},
        {"armed_timeout_ms", config.armed_timeout_ms},
        {"pointer_gain_pixels", config.pointer_gain_pixels},
        {"pointer_smoothing", config.pointer_smoothing},
        {"pointer_deadzone_normalized", config.pointer_deadzone_normalized},
        {"pointer_deadzone_start_multiplier", config.pointer_deadzone_start_multiplier},
        {"pointer_one_euro_min_cutoff", config.pointer_one_euro_min_cutoff},
        {"pointer_one_euro_beta", config.pointer_one_euro_beta},
        {"pointer_one_euro_derivative_cutoff",
         config.pointer_one_euro_derivative_cutoff},
        {"pointer_outlier_velocity_normalized_per_second",
         config.pointer_outlier_velocity_normalized_per_second},
        {"pointer_tip_weight", config.pointer_tip_weight},
        {"pointer_hand_lock_radius_normalized",
         config.pointer_hand_lock_radius_normalized},
        {"pointer_hand_loss_grace_ms", config.pointer_hand_loss_grace_ms},
        {"max_pointer_step_pixels", config.max_pointer_step_pixels},
        {"invert_pointer_x", config.invert_pointer_x},
        {"invert_pointer_y", config.invert_pointer_y},
        {"bindings", std::move(bindings)}
    };
}

bool PersistPhysicalGestureControlConfig(std::string& error) {
    error.clear();
    try {
        nlohmann::json pending;
        pending["physical_interaction"]["gesture_control"] =
            SerializePhysicalGestureControlConfig();
        (void)Settings::saveRuntimeAiConfig(pending);
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

}}} // namespace GRIM::Perception::Physical
