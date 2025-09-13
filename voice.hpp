#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// ✅ Forward declare at global scope
struct whisper_context;

namespace Voice {

// ---------------- State ----------------
struct State {
    ::whisper_context* ctx = nullptr;   // ✅ Whisper model context
    float lastSegmentConfidence = -1.0f;
    int minSpeechMs = 500;
    int minSilenceMs = 1200;
    int inputDeviceIndex = -1;
    double envLevel = 0.0;
};

// Extern instance (defined in voice.cpp)
extern State g_state;

// ---------------- API ----------------

// Initialize Whisper model
bool initWhisper(const std::string& modelName = "small", std::string* err = nullptr);

// Record microphone, transcribe with Whisper, return text
std::string runVoiceDemo(nlohmann::json& longTermMemory);

// Speak text aloud (tries online TTS first, then offline fallback)
bool speakText(const std::string& text, bool preferOnline = true);

} // namespace Voice
