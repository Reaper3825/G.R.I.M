#pragma once
#include <string>

namespace Voice {
    // =========================================================
    // Lifecycle
    // =========================================================

    // 🔹 Initialize TTS (load config, start persistent Coqui bridge)
    bool initTTS();

    // 🔹 Shutdown TTS (send exit to bridge, cleanup)
    void shutdownTTS();

    // 🔹 Query if TTS bridge is ready (handshake + model loaded)
    bool isReady();

    // =========================================================
    // Audio
    // =========================================================

    // 🔹 Play back an audio file (async, non-blocking)
    void playAudio(const std::string& path);

    // =========================================================
    // Coqui TTS Bridge (persistent mode)
    // =========================================================

    // 🔹 Send text → receive generated .wav file path from bridge
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);

    // =========================================================
    // High-level
    // =========================================================

    // 🔹 Unified entry point (routes by category → engine)
    void speak(const std::string& text, const std::string& category);
}
