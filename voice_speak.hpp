#pragma once
#include <string>

// =========================================================
// Public entry point
// =========================================================
void speak(const std::string& text, const std::string& category = "routine");

// =========================================================
// Local speech (SAPI / Piper / say)
// =========================================================
bool speakLocal(const std::string& text, const std::string& voiceModel);

// =========================================================
// Cloud speech (OpenAI / ElevenLabs / Azure)
// For now: stub in voice_speak.cpp
// =========================================================
bool speakCloud(const std::string& text, const std::string& engine);

// =========================================================
// Utility: play a WAV file with SFML
// =========================================================
void playAudio(const std::string& path);
