#pragma once

#include "tts_provider.hpp"

#include <memory>
#include <string>

namespace Voice {

struct CosyVoiceProviderConfig {
    std::string python_executable;
    std::string repository_path;
    std::string model_path;
    std::string bridge_script;
    std::string output_directory;
    std::string reference_audio_path;
    std::string reference_text;
    int startup_timeout_ms = 180000;
    int synthesis_timeout_ms = 120000;
};

std::unique_ptr<ITTSProvider> createCosyVoiceProvider(
    CosyVoiceProviderConfig config);

} // namespace Voice
