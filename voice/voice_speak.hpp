#pragma once
#include <string>

namespace Voice {
    bool initTTS();
    void shutdownTTS();
    bool isReady();
    bool isPlaying();

    // Queue management
    void initQueue();
    void shutdownQueue();

    // Speech
    void speak(const std::string& text, const std::string& category);
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);
    void playAudio(const std::string& path);
}
