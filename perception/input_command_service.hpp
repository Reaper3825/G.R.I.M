#pragma once
#include "core/plugin_api.hpp"
#include "commands/command_result.hpp"
#include <string>

namespace GRIM::Perception {
// Host-owned input/context boundary; no OpenCV, input device, or live manager crosses the ABI.
// Preserves existing command parsing, effects, and complete results. Never redispatches a tool.
CommandResult ExecuteInputCommandService(GrimInputOperation operation, const std::string& arguments);
}
