#pragma once
#include <string>

namespace Voice {
    // =========================================================
    // Voice interface (public API)
    // =========================================================

    // 🔹 Play back an audio file (async, non-blocking)
    void playAudio(const std::string& path);

    // 🔹 Local speech synthesis (SAPI on Windows, say on macOS, Piper on Linux)
    bool speakLocal(const std::string& text, const std::string& voiceModel);

    // 🔹 Cloud speech synthesis (stub for now)
    bool speakCloud(const std::string& text, const std::string& engine);

    // 🔹 Unified entry point (routes by category → engine)
    void speak(const std::string& text, const std::string& category);

    // 🔹 Simplified helper (legacy mode: pick local/cloud)
    bool speakText(const std::string& text, bool preferOnline = true);

    // =========================================================
    // Coqui TTS Persistent Bridge
    // =========================================================

    // 🔹 Start the Python bridge (load model once into memory)
    bool initTTS();

    // 🔹 Stop the Python bridge (terminate process, close pipes)
    void shutdownTTS();

    // 🔹 Send text → receive .wav file path from bridge
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);
}
