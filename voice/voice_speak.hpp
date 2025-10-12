#pragma once
#include <string>
#include <memory>
#include <SFML/Audio.hpp>

namespace Voice {
    bool initTTS();
    void shutdownTTS();
    bool isReady();
    bool isPlaying();
    bool isSpeaking();
    // Queue management
    void initQueue();
    void shutdownQueue();

    // Speech
    void speak(const std::string& text, const std::string& category);
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed);
    void playAudio(const std::string& path);
    extern std::unique_ptr<sf::Sound> g_activeSound;
    extern std::shared_ptr<sf::SoundBuffer> g_activeBuffer;
}
