#pragma once
#include <string>
#include <nlohmann/json_fwd.hpp>
#include "whisper.h"  // pull in the real whisper_context

namespace Voice {

    struct State {
        ::whisper_context* ctx = nullptr;  // global whisper context
        int inputDeviceIndex = -1;
        int minSpeechMs = 500;
        int minSilenceMs = 1200;
    };

    extern State g_state;

    // =========================================================
    // Whisper setup
    // =========================================================
    bool initWhisper(const std::string& modelName, std::string* err = nullptr);

    // =========================================================
    // Voice input (speech → text)
    // Loads thresholds from aiConfig directly
    // =========================================================
    std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory);

    // =========================================================
    // Cleanup
    // =========================================================
    void shutdown();

} // namespace Voice
