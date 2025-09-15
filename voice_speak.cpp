#include "voice_speak.hpp"
#include "resources.hpp"
#include "ai.hpp"   // extern aiConfig

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <SFML/Audio.hpp>

namespace fs = std::filesystem;

// =========================================================
// Windows SAPI headers
// =========================================================
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <objbase.h>
#include <sapi.h>
#include <sphelper.h>
#endif

// =========================================================
// Helpers
// =========================================================
static nlohmann::json getVoiceConfig() {
    // Bootstrap guarantees aiConfig["voice"] exists and is valid
    return aiConfig["voice"];
}

// =========================================================
// Audio Playback (SFML)
// =========================================================
void playAudio(const std::string& path) {
    static sf::Sound sound;
    static sf::SoundBuffer buffer;

    if (!buffer.loadFromFile(path)) {
        std::cerr << "[Voice] Failed to load audio: " << path << "\n";
        return;
    }

    sound.setBuffer(buffer);
    sound.play();

    while (sound.getStatus() == sf::Sound::Playing) {
        sf::sleep(sf::milliseconds(100));
    }
}

// =========================================================
// Local Speech (cross-platform)
// =========================================================
bool speakLocal(const std::string& text, const std::string& voiceModel) {
#ifdef _WIN32
    (void)voiceModel; // not used on Windows
    std::wstring wtext(text.begin(), text.end());
    ISpVoice* pVoice = nullptr;

    if (FAILED(::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED))) return false;

    HRESULT hr = CoCreateInstance(
        CLSID_SpVoice, NULL, CLSCTX_ALL,
        IID_ISpVoice, (void**)&pVoice);

    if (SUCCEEDED(hr)) {
        pVoice->Speak(wtext.c_str(), SPF_DEFAULT, NULL);
        pVoice->Release();
    }
    ::CoUninitialize();
    return SUCCEEDED(hr);

#elif __APPLE__
    std::string safeText = text;
    size_t pos = 0;
    while ((pos = safeText.find("\"", pos)) != std::string::npos) {
        safeText.replace(pos, 1, "\\\"");
        pos += 2;
    }
    std::string cmd = "say \"" + safeText + "\"";
    return system(cmd.c_str()) == 0;

#else
    std::string outPath = "resources/cache/tts/local.wav";
    fs::create_directories("resources/cache/tts");

    std::string safeText = text;
    size_t pos = 0;
    while ((pos = safeText.find("\"", pos)) != std::string::npos) {
        safeText.replace(pos, 1, "\\\"");
        pos += 2;
    }

    std::string cmd = "echo \"" + safeText + "\" | piper --model voices/" + voiceModel + " --output_file " + outPath;
    int res = std::system(cmd.c_str());
    if (res != 0) {
        std::cerr << "[Voice] Piper failed\n";
        return false;
    }
    playAudio(outPath);
    return true;
#endif
}

// =========================================================
// Cloud Speech (stub for now)
// =========================================================
bool speakCloud(const std::string& text, const std::string& engine) {
    std::cerr << "[Voice] Cloud TTS (" << engine << ") not implemented yet.\n";
    (void)text; (void)engine;
    return false;
}

// =========================================================
// Public Entry
// =========================================================
void speak(const std::string& text, const std::string& category) {
    auto cfg = getVoiceConfig();

    std::string rule        = cfg["rules"].value(category, "local");
    std::string mode        = cfg.value("mode", "local");
    std::string localEngine = cfg.value("local_engine", "en_US-amy-medium.onnx");
    std::string cloudEngine = cfg.value("cloud_engine", "openai");

    std::cout << "[DEBUG][Voice] speak(\"" << text << "\", category=" << category
              << ") → mode=" << mode << " rule=" << rule
              << " local=" << localEngine << " cloud=" << cloudEngine << "\n";

    bool success = false;

    if (mode == "local") {
        success = speakLocal(text, localEngine);
    } else if (mode == "cloud") {
        success = speakCloud(text, cloudEngine);
    } else if (mode == "hybrid") {
        // Try cloud first, fallback to local
        success = speakCloud(text, cloudEngine);
        if (!success) {
            std::cerr << "[Voice] Cloud failed, falling back to local.\n";
            success = speakLocal(text, localEngine);
        }
    }

    if (!success) {
        std::cerr << "[Voice] Speech failed for text: " << text << "\n";
    }
}
