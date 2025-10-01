#pragma once
#include <string>
#include <nlohmann/json.hpp>

// Forward declare
struct whisper_context;

namespace Voice {
    struct State {
        whisper_context* ctx = nullptr;
        int minSpeechMs = 0;
        int minSilenceMs = 0;
        int inputDeviceIndex = -1;
    };

    extern State g_state;

    // Already present
    std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory);
    void shutdown();

    // 🔹 Add this:
    whisper_context* getWhisperContext();
}
