#pragma once
#include <string>
#include <atomic>

namespace Voice {
    extern std::atomic<bool> g_isSpeaking;

    bool initTTS();
    void shutdownTTS();
    bool isReady();

    void speak(const std::string& text, const std::string& category);
    void playAudio(const std::string& path);
    bool isPlaying();
    bool isSpeaking();

    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);

    void initQueue();
    void shutdownQueue();
}
