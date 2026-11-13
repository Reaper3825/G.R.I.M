#pragma once
#include "core/plugin_api.hpp"
#include "commands/command_result.hpp"
#include <string>
#include <nlohmann/json.hpp>

namespace GRIM::Perception {
// Host-owned input/context boundary; no OpenCV, input device, or live manager crosses the ABI.
// Executes parsed device operations only; false means live context unavailable.
bool ExecuteInputPrimitive(const nlohmann::json& request);
// Retained legacy handlers; click_on still uses this command boundary.
CommandResult ExecuteInputCommandService(GrimInputOperation operation, const std::string& arguments);
}
