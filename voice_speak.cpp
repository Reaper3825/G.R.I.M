#include "voice_speak.hpp"
#include "resources.hpp"
#include "ai.hpp"

#include <SFML/Audio.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <mutex>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

#ifdef _WIN32
    #include <objbase.h>
    #include <windows.h>
    #include <sapi.h>
    #include <sphelper.h>
#endif

namespace Voice {

// =========================================================
// Persistent state
// =========================================================
static bool g_ttsReady = false;

namespace {
    std::mutex g_audioMutex;

    // Short sound effects (if you ever need them)
    std::vector<std::unique_ptr<sf::SoundBuffer>> g_buffers;
    std::vector<std::unique_ptr<sf::Sound>> g_sounds;

    // Long audio streams (TTS / music / Coqui)
    std::vector<std::unique_ptr<sf::Music>> g_music;

    std::string timestampNow() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
    #ifdef _WIN32
        localtime_s(&tm, &t);
    #else
        localtime_r(&t, &tm);
    #endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%H:%M:%S");
        return oss.str();
    }
}

static nlohmann::json getVoiceConfig() {
    return aiConfig["voice"];
}


// =========================================================
// Audio Playback (SFML async)
// =========================================================
void playAudio(const std::string& path) {
    std::thread([path]() {
        // Probe duration
        sf::SoundBuffer probe;
        if (!probe.loadFromFile(path)) {
            std::cerr << "[Voice] Failed to open audio: " << path << "\n";
            return;
        }

        float duration = probe.getDuration().asSeconds();
        std::cout << "[Voice][" << timestampNow() << "] Preparing to play "
                  << path << " (duration " << duration << "s)\n";

        if (duration < 10.0f) {
            // 🔹 Short clip → Sound
            auto buffer = std::make_unique<sf::SoundBuffer>(probe);
            auto sound  = std::make_unique<sf::Sound>(*buffer);

            {
                std::lock_guard<std::mutex> lock(g_audioMutex);
                g_buffers.push_back(std::move(buffer));
                g_sounds.push_back(std::move(sound));
            }

            sf::Sound* s = g_sounds.back().get();
            s->play();

            while (s->getStatus() == sf::Sound::Playing) {
                sf::sleep(sf::milliseconds(100));
            }
        } else {
            // 🔹 Long clip → Music
            auto music = std::make_unique<sf::Music>();
            if (!music->openFromFile(path)) {
                std::cerr << "[Voice] Failed to stream audio: " << path << "\n";
                return;
            }

            {
                std::lock_guard<std::mutex> lock(g_audioMutex);
                g_music.push_back(std::move(music));
            }

            sf::Music* m = g_music.back().get();
            m->play();

            while (m->getStatus() == sf::Music::Playing) {
                sf::sleep(sf::milliseconds(100));
            }
        }

        std::cout << "[Voice][" << timestampNow() << "] Finished playback\n";
    }).detach();
}



// =========================================================
// Windows SAPI Local Speech
// =========================================================
bool speakLocal(const std::string& text, const std::string& voiceName) {
#ifdef _WIN32
    std::cout << "[Voice][SAPI] SpeakLocal called (voice=" << voiceName << ")\n";

    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "[Voice][SAPI] CoInitializeEx failed: 0x" << std::hex << hr << "\n";
        return false;
    }

    ISpVoice* pVoice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (FAILED(hr) || !pVoice) {
        std::cerr << "[Voice][SAPI] Failed to create ISpVoice\n";
        CoUninitialize();
        return false;
    }

    // Select specific voice if requested
    if (!voiceName.empty()) {
        ISpObjectToken* pToken = nullptr;
        IEnumSpObjectTokens* pEnum = nullptr;
        ULONG count = 0;
        if (SUCCEEDED(SpEnumTokens(SPCAT_VOICES, NULL, NULL, &pEnum)) && pEnum) {
            while (pEnum->Next(1, &pToken, &count) == S_OK) {
                WCHAR* desc = nullptr;
                SpGetDescription(pToken, &desc);
                if (desc) {
                    std::wstring wdesc(desc);
                    CoTaskMemFree(desc);
                    if (wdesc.find(std::wstring(voiceName.begin(), voiceName.end())) != std::wstring::npos) {
                        std::wcout << L"[Voice][SAPI] Selecting voice: " << wdesc << L"\n";
                        pVoice->SetVoice(pToken);
                        break;
                    }
                }
                pToken->Release();
            }
            pEnum->Release();
        }
    }

    // Proper UTF-8 → UTF-16 conversion (no truncation)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), NULL, 0);
    std::wstring wtext(wlen, 0);
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), (int)text.size(), &wtext[0], wlen);

    hr = pVoice->Speak(wtext.c_str(), SPF_ASYNC, NULL);
    if (FAILED(hr)) {
        std::cerr << "[Voice][SAPI] Speak() failed\n";
        pVoice->Release();
        CoUninitialize();
        return false;
    }

    pVoice->WaitUntilDone(INFINITE);
    pVoice->Release();
    CoUninitialize();
    return true;
#else
    (void)text; (void)voiceName;
    return false;
#endif
}

// =========================================================
// Cloud/Coqui speech (stub for now)
// =========================================================
bool speakCloud(const std::string& text, const std::string& engine) {
    std::cerr << "[Voice] speakCloud not implemented (engine=" << engine << ")\n";
    (void)text;
    return false;
}

// =========================================================
// Unified API
// =========================================================
bool initTTS() {
    std::cout << "[Voice] initTTS() initializing...\n";
    g_ttsReady = true;
    return true;
}

void shutdownTTS() {
    std::cout << "[Voice] shutdownTTS() cleaning up...\n";
    g_ttsReady = false;
}

void speak(const std::string& engine, const std::string& text) {
    if (!g_ttsReady) {
        std::cerr << "[Voice] TTS not initialized!\n";
        return;
    }

    auto cfg = getVoiceConfig();
    std::string voiceName = cfg.value("voice", "");

    if (engine == "sapi") {
        speakLocal(text, voiceName);
    } else if (engine == "coqui") {
        if (!speakCloud(text, engine)) {
            std::cerr << "[Voice] Coqui fallback failed → using local SAPI.\n";
            speakLocal(text, voiceName);
        }
    } else {
        std::cerr << "[Voice] Unknown engine=" << engine << " → defaulting to SAPI.\n";
        speakLocal(text, voiceName);
    }
}

} // namespace Voice
