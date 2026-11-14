#pragma once
#include <string>
#include <atomic>
#include "tts_provider.hpp"

namespace Voice {
    extern std::atomic<bool> g_isSpeaking;

    // Voice prosody parameters from EmotionPresentationController
    struct VoiceParams {
        float pitch    = 1.0f;  // multiplier (1.0 = neutral)
        float rate     = 1.0f;  // multiplier (1.0 = neutral)
        float emphasis = 0.5f;  // 0..1 range
    };

    bool initTTS();
    void shutdownTTS();
    bool isReady();
    const char* activeTTSProviderId();
    TTSProviderState activeTTSProviderState();
    TTSProviderCapabilities activeTTSProviderCapabilities();
    TTSSynthesisResult synthesize(const TTSSynthesisRequest& request);

    void speak(const std::string& text, const std::string& category);
    void speak(const std::string& text, const std::string& category, const VoiceParams& params);
    void playAudio(const std::string& path);
    bool isPlaying();
    bool isSpeaking();

    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed,
                           double pitch = 1.0);

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
    void setSpeaker(const std::string& speaker);  // ? NEW: Update speaker dynamically
    std::string getSpeaker();                      // ? NEW: Get current speaker
}
