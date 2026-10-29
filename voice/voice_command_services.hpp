#pragma once
#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>

namespace GRIM::VoiceCommandServices {
nlohmann::json execute(GrimVoiceOperation operation, const nlohmann::json& request);
}
