#pragma once
#include <nlohmann/json.hpp>

namespace GRIM::VoiceCommandServices {
nlohmann::json listSapiVoices();
nlohmann::json sapiOutputDevice();
// Preserve the existing command paths; no new configuration/relocation policy.
nlohmann::json paths();
}
