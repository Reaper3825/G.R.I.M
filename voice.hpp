#pragma once
#include <string>
#include "whisper.h"

// Global Whisper context (shared across GRIM)
extern struct whisper_context *g_whisperCtx;

// Initialize Whisper globally (auto-fallback base.en → small)
bool initWhisper();

// Legacy demo: short recording and transcription
std::string runVoiceDemo(const std::string &modelPath);
