#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "whisper.h"  // pull in the real whisper_context

namespace Voice {

    struct State {
        ::whisper_context* ctx = nullptr;  // use global scope type
        int inputDeviceIndex = -1;
        int minSpeechMs = 500;
        int minSilenceMs = 1200;
    };

    extern State g_state;

    // Whisper setup
    bool initWhisper(const std::string& modelName, std::string* err = nullptr);

    // Voice input (speech → text), pulls thresholds directly from aiConfig
    std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory);

    // Voice output (text → speech)
    bool speakText(const std::string& text, bool preferOnline = true);

} // namespace Voice
