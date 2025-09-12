#pragma once
#include <string>
#include "whisper.h"
#include <nlohmann/json.hpp>


// Global Whisper context (shared across GRIM)
extern struct whisper_context *g_whisperCtx;

// Initialize Whisper globally (auto-fallback base.en → small)
bool initWhisper();

// Legacy demo: short recording and transcription
std::string runVoiceDemo(const std::string &modelPath,
                         nlohmann::json& longTermMemory);
