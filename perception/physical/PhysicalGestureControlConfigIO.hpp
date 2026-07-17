#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// Applies physical_interaction.gesture_control from an already-loaded runtime
// document. A missing section is valid and leaves the safe built-in defaults.
bool ApplyPhysicalGestureControlConfigFromRuntime(
    const nlohmann::json& runtime_config, std::string& error);

// Returns the complete gesture_control object (not the outer runtime document).
nlohmann::json SerializePhysicalGestureControlConfig();

// Persists the current controller configuration through the canonical settings
// merge path. No model, dependency, or network operation is performed.
bool PersistPhysicalGestureControlConfig(std::string& error);

}}} // namespace GRIM::Perception::Physical
