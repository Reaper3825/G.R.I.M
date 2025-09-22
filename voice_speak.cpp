#include "voice_speak.hpp"

#include <SFML/Audio.hpp>
#include <filesystem>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <mutex>
#include <algorithm>
#include <ctime>

namespace fs = std::filesystem;

// =========================================================
// 🔹 Global state for audio playback
// =========================================================
static std::mutex g_audioMutex;
static std::vector<std::shared_ptr<sf::SoundBuffer>> g_buffers;
static std::vector<std::unique_ptr<sf::Sound>> g_sounds;
static std::vector<std::unique_ptr<sf::Music>> g_music;

// =========================================================
// 🔹 Helper: timestamp for logs (safe version)
// =========================================================
static std::string timestampNow() {
    auto t = std::time(nullptr);
    struct tm buf {};
    localtime_s(&buf, &t); // ✅ safe MSVC version

    char out[32];
    std::strftime(out, sizeof(out), "%H:%M:%S", &buf);
    return out;
}

namespace Voice {

    // =====================================================
    // 🔹 Play back an audio file (async, non-blocking)
    // =====================================================
    void playAudio(const std::string& path) {
        std::thread([path]() {
            // --------------------------
            // Check file exists & ready
            // --------------------------
            for (int i = 0; i < 50; i++) { // ~5s max wait
                if (fs::exists(path) && fs::file_size(path) > 44) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            if (!fs::exists(path)) {
                std::cerr << "[Voice][" << timestampNow() << "] ERROR: File not found: "
                          << path << "\n";
                return;
            }

            auto size = fs::file_size(path);
            if (size <= 44) {
                std::cerr << "[Voice][" << timestampNow() << "] ERROR: Invalid/empty WAV file: "
                          << path << " (" << size << " bytes)\n";
                return;
            }

            // --------------------------
            // Probe duration
            // --------------------------
            sf::SoundBuffer probe;
            if (!probe.loadFromFile(path)) {
                std::cerr << "[Voice][" << timestampNow() << "] ERROR: Failed to open audio: "
                          << path << "\n";
                return;
            }

            float duration = probe.getDuration().asSeconds();
            std::cout << "[Voice][" << timestampNow() << "] Preparing to play "
                      << path << " (duration " << duration << "s, size "
                      << size << " bytes)\n";

            if (duration < 10.0f) {
                // 🔹 Short clip → use Sound
                auto buffer = std::make_shared<sf::SoundBuffer>(probe);
                auto sound  = std::make_unique<sf::Sound>(*buffer);

                {
                    std::lock_guard<std::mutex> lock(g_audioMutex);
                    g_buffers.push_back(buffer);
                    g_sounds.push_back(std::move(sound));
                }

                sf::Sound* s = g_sounds.back().get();
                s->play();
                std::cout << "[Voice][" << timestampNow() << "] Started short-clip playback\n";

            } else {
                // 🔹 Long clip → stream with Music
                auto music = std::make_unique<sf::Music>();
                if (!music->openFromFile(path)) {
                    std::cerr << "[Voice][" << timestampNow() << "] ERROR: Failed to stream: "
                              << path << "\n";
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(g_audioMutex);
                    g_music.push_back(std::move(music));
                }

                sf::Music* m = g_music.back().get();
                m->play();
                std::cout << "[Voice][" << timestampNow() << "] Started streaming playback\n";
            }

            // --------------------------
            // Background cleanup loop
            // --------------------------
            for (;;) {
                {
                    std::lock_guard<std::mutex> lock(g_audioMutex);

                    g_sounds.erase(
                        std::remove_if(g_sounds.begin(), g_sounds.end(),
                            [](auto& s){ return s->getStatus() == sf::SoundSource::Status::Stopped; }),
                        g_sounds.end()
                    );

                    g_music.erase(
                        std::remove_if(g_music.begin(), g_music.end(),
                            [](auto& m){ return m->getStatus() == sf::SoundSource::Status::Stopped; }),
                        g_music.end()
                    );
                }

                // exit when no active sounds
                {
                    std::lock_guard<std::mutex> lock(g_audioMutex);
                    if (g_sounds.empty() && g_music.empty()) break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }

            std::cout << "[Voice][" << timestampNow() << "] Finished playback\n";
        }).detach();
    }

} // namespace Voice

namespace Voice {

    bool initTTS() {
        std::cout << "[Voice] initTTS() stub called" << std::endl;
        return true; // ✅ success by default
    }

    void shutdownTTS() {
        std::cout << "[Voice] shutdownTTS() stub called" << std::endl;
    }

    void speak(const std::string& text, const std::string& category) {
        std::cout << "[Voice] speak() stub called: text=\"" << text
                  << "\", category=\"" << category << "\"" << std::endl;
        // You can call playAudio() here if you want actual audio
        // playAudio("some_file.wav");
    }

    std::string coquiSpeak(const std::string& text,
                       const std::string& speaker,
                       double speed) {
    namespace fs = std::filesystem;

    // 🔹 Pick output file
    std::string outDir = "D:/G.R.I.M/resources/tts_out";
    fs::create_directories(outDir);

    std::string outFile = outDir + "/" + std::to_string(std::time(nullptr)) + ".wav";

    // 🔹 Build python command
    std::string cmd =
        "python D:/G.R.I.M/resources/python/coqui_bridge.py "
        "\"" + text + "\" "
        "--speaker " + speaker +
        " --speed " + std::to_string(speed) +
        " --out \"" + outFile + "\"";

    std::cout << "[Voice][Coqui] Running: " << cmd << std::endl;

    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::cerr << "[Voice][Coqui] ERROR: Bridge failed with code " << rc << std::endl;
        return {};
    }

    // 🔹 Wait for output file
    for (int i = 0; i < 50; i++) {
        if (fs::exists(outFile) && fs::file_size(outFile) > 44) {
            std::cout << "[Voice][Coqui] File ready: " << outFile << std::endl;
            return outFile;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cerr << "[Voice][Coqui] ERROR: No valid audio file created" << std::endl;
    return {};
}


} // namespace Voice
