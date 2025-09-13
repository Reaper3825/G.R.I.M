#pragma once
#include <string>
#include "whisper.h"
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Global Whisper context (declared here, defined in voice.cpp)
// ------------------------------------------------------------
extern struct whisper_context* g_whisperCtx;
extern float g_lastSegmentConfidence;
// Silence timeout in milliseconds (configurable from ai_config.json)
extern int g_silenceTimeoutMs;


// ------------------------------------------------------------
// Function declarations (implemented in voice.cpp)
// ------------------------------------------------------------

// Initialize Whisper globally (auto-fallback base.en → small)
bool initWhisper();

// Legacy demo: short recording and transcription
std::string runVoiceDemo(const std::string& modelPath,
                         nlohmann::json& longTermMemory);

// Optional simple demo version (no args)
void runVoiceDemo();
