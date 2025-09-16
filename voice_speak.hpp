#pragma once
#include <string>

// =========================================================
// Voice interface (public API)
// =========================================================

// 🔹 Play back an audio file (async, non-blocking)
void playAudio(const std::string& path);

// 🔹 Local speech synthesis (SAPI on Windows, say on macOS, Piper on Linux)
bool speakLocal(const std::string& text, const std::string& voiceModel);

// 🔹 Cloud speech synthesis (stub for now)
bool speakCloud(const std::string& text, const std::string& engine);

// 🔹 Unified entry point (decides local/cloud/hybrid based on ai_config.json)
void speak(const std::string& text, const std::string& category);
