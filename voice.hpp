#pragma once
#include <string>
#include "whisper.h"
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Global Whisper context (declared here, defined in voice.cpp)
// ------------------------------------------------------------
extern struct whisper_context* g_whisperCtx;
extern float g_lastSegmentConfidence;

// Silence / timing config values (loaded from ai_config.json)
extern double g_silenceThreshold;   // energy threshold
extern int g_silenceTimeoutMs;      // ms before final cutoff
extern int g_minSpeechMs;           // minimum speech duration
extern int g_minSilenceMs;          // minimum silence duration

// ------------------------------------------------------------
// Function declarations (implemented in voice.cpp)
// ------------------------------------------------------------

// Initialize Whisper globally (auto-fallback base.en → small)
bool initWhisper();
bool initWhisper(const std::string& modelName);

// Voice demo: recording and transcription
std::string runVoiceDemo(const std::string& modelPath,
                         nlohmann::json& longTermMemory);
