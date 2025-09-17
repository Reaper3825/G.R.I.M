#include "voice_speak.hpp"
#include "resources.hpp"
#include "ai.hpp"   // extern aiConfig

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <thread>
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
    return aiConfig["voice"];
}

namespace Voice {

// =========================================================
// Audio Playback (SFML, async)
// =========================================================
void playAudio(const std::string& path) {
    std::thread([path]() {
        static sf::Sound sound;
        static sf::SoundBuffer buffer;

        if (!buffer.loadFromFile(path)) {
            std::cerr << "[Voice] Failed to load audio: " << path << "\n";
            return;
        }

        sound.setBuffer(buffer);
        sound.play();

        std::cout << "[Voice] Playing audio (non-blocking): " << path << "\n";

        while (sound.getStatus() == sf::Sound::Playing) {
            sf::sleep(sf::milliseconds(100));
        }
    }).detach();
}

// =========================================================
// Local Speech (cross-platform)
// =========================================================
bool speakLocal(const std::string& text, const std::string& voiceModel) {
#ifdef _WIN32
    std::wstring wtext(text.begin(), text.end());
    ISpVoice* pVoice = nullptr;

    HRESULT hrInit = ::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hrInit)) {
        std::cerr << "[Voice] CoInitializeEx failed, HRESULT=0x"
                  << std::hex << hrInit << std::dec << "\n";
        return false;
    }

    HRESULT hr = CoCreateInstance(
        CLSID_SpVoice, NULL, CLSCTX_ALL,
        IID_ISpVoice, (void**)&pVoice);

    if (FAILED(hr)) {
        std::cerr << "[Voice] SAPI CoCreateInstance failed, HRESULT=0x"
                  << std::hex << hr << std::dec << "\n";
        ::CoUninitialize();
        return false;
    }

    // 🔹 Force output to default audio device
    hr = pVoice->SetOutput(NULL, TRUE);
    if (FAILED(hr)) {
        std::cerr << "[Voice] Failed to bind voice to default audio device, HRESULT=0x"
                  << std::hex << hr << std::dec << "\n";
    }

    // 🔹 Enumerate installed voices
    IEnumSpObjectTokens* cpEnum = nullptr;
    hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &cpEnum);
    if (SUCCEEDED(hr) && cpEnum) {
        ULONG count = 0;
        cpEnum->GetCount(&count);
        std::cout << "[Voice] Installed voices: " << count << "\n";

        for (ULONG i = 0; i < count; i++) {
            ISpObjectToken* cpVoiceToken = nullptr;
            if (SUCCEEDED(cpEnum->Next(1, &cpVoiceToken, NULL))) {
                LPWSTR description = nullptr;
                SpGetDescription(cpVoiceToken, &description);

                std::wcout << L"[Voice] [" << i << L"] " << description << "\n";

                if (i == 0) { // pick the first by default
                    hr = pVoice->SetVoice(cpVoiceToken);
                    if (SUCCEEDED(hr)) {
                        std::wcout << L"[Voice] Using voice: " << description << "\n";
                    } else {
                        std::cerr << "[Voice] Failed to set voice index " << i << "\n";
                    }
                }

                ::CoTaskMemFree(description);
                cpVoiceToken->Release();
            }
        }
        cpEnum->Release();
    } else {
        std::cerr << "[Voice] No voices found or SpEnumTokens failed.\n";
    }

    // 🔹 Speak synchronously
    hr = pVoice->Speak(wtext.c_str(), SPF_DEFAULT, NULL);
    if (FAILED(hr)) {
        std::cerr << "[Voice] SAPI Speak() failed, HRESULT=0x"
                  << std::hex << hr << std::dec << "\n";
        pVoice->Release();
        ::CoUninitialize();
        return false;
    }

    pVoice->Release();
    ::CoUninitialize();
    return true;

#elif __APPLE__
    std::string safeText = text;
    size_t pos = 0;
    while ((pos = safeText.find("\"", pos)) != std::string::npos) {
        safeText.replace(pos, 1, "\\\"");
        pos += 2;
    }
    std::string cmd = "say \"" + safeText + "\" &";
    int res = system(cmd.c_str());
    if (res != 0) {
        std::cerr << "[Voice] macOS say command failed, code=" << res << "\n";
        return false;
    }
    return true;

#else
    fs::create_directories("resources/cache/tts");
    std::string outPath = "resources/cache/tts/local.wav";

    std::string safeText = text;
    size_t pos = 0;
    while ((pos = safeText.find("\"", pos)) != std::string::npos) {
        safeText.replace(pos, 1, "\\\"");
        pos += 2;
    }

    std::string cmd = "echo \"" + safeText + "\" | piper --model voices/"
                    + voiceModel + " --output_file " + outPath;
    int res = std::system(cmd.c_str());
    if (res != 0) {
        std::cerr << "[Voice] Piper failed with code=" << res << "\n";
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
// Unified Helper: speakText
// =========================================================
bool speakText(const std::string& text, bool preferOnline) {
    auto cfg = getVoiceConfig();

    std::string mode        = cfg.value("mode", "hybrid");
    std::string localEngine = cfg.value("local_engine", "en_US-amy-medium.onnx");
    std::string cloudEngine = cfg.value("cloud_engine", "openai");

    std::cout << "[DEBUG][Voice] speakText(\"" << text
              << "\", preferOnline=" << (preferOnline ? "true" : "false")
              << ") → mode=" << mode
              << " local=" << localEngine
              << " cloud=" << cloudEngine << "\n";

    bool success = false;

    if (preferOnline && mode != "local") {
        success = speakCloud(text, cloudEngine);
    }
    if (!success) {
        success = speakLocal(text, localEngine);
    }

    if (!success) {
        std::cerr << "[Voice] Speech failed for text: \"" << text << "\"\n";
        return false;
    }
    return true;
}

// =========================================================
// Unified Entry Point: speak (category-based)
// =========================================================
void speak(const std::string& text, const std::string& category) {
    (void)category; // category currently not used
    speakText(text, false);
}

} // namespace Voice
