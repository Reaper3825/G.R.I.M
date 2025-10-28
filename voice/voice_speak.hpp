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
    
    // ? NEW: Pre-cache management
    void preCacheCommonPhrases();
    void initPreCache();

    // Audio state and playback integration
    // XTTS v2 Utility Functions
    bool isXTTSv2Enabled();
    void setLanguage(const std::string& lang);
    std::string getLanguage();
}
