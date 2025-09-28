#pragma once

#include <string>

// 🔹 Add popup control API so Voice can notify activity
#include "popup_ui/popup_ui.hpp"

namespace Voice {
    // Initialize the TTS system (Coqui / SAPI bridge)
    bool initTTS();


    // Shut down the TTS system
    void shutdownTTS();


    // Query if TTS is ready
    bool isReady();


    // Play an audio file directly
    void playAudio(const std::string& path);

    
    // Send text to Coqui TTS (returns wav path if successful)
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);


    // High-level speak function (routes to engine)
    void speak(const std::string& text,
               const std::string& category);

    // Returns true if any voice audio is currently playing
    bool isPlaying();
}
