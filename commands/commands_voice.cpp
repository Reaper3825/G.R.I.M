#if defined(_WIN32)
// ---------------------------------------------------------
// Windows + SAPI includes
// ---------------------------------------------------------
    #define NOMINMAX              // prevent min/max macros
    #define WIN32_LEAN_AND_MEAN   // strip rarely-used APIs from windows.h

    #include <sapi.h>             // ISpVoice, ISpStream
    #include <sphelper.h>    

    // Link against required libs
    #pragma comment(lib, "sapi.lib")
    #pragma comment(lib, "ole32.lib")
    #pragma comment(lib, "oleaut32.lib")
    #pragma comment(lib, "shlwapi.lib")

    // Cleanup macro pollution from Windows headers
    #undef ERROR
    #undef min
    #undef max
#endif

// ---------------------------------------------------------
// GRIM project includes
// ---------------------------------------------------------
#include "commands_voice.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "commands_core.hpp"
#include "voice_speak.hpp"
#include "resources.hpp" 
#include "nlp.hpp"   // globals: history, timers, longTermMemory, g_nlp

// ---------------------------------------------------------
// SFML
// ---------------------------------------------------------
#include <SFML/Audio.hpp>

// ---------------------------------------------------------
// Standard headers
// ---------------------------------------------------------
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>

// ---------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------
namespace {
    bool playWavFile(const std::string& wavPath) {
        // Keep both buffers and sounds alive until playback finishes
        static std::vector<std::unique_ptr<sf::SoundBuffer>> buffers;
        static std::vector<sf::Sound> sounds;

        auto buffer = std::make_unique<sf::SoundBuffer>();
        if (!buffer->loadFromFile(wavPath)) {
            std::cerr << "[Voice][Error] Failed to load audio: " << wavPath << std::endl;
            return false;
        }

        sf::Sound sound(*buffer);
        sound.play();

        // Push into static storage
        sounds.push_back(std::move(sound));
        buffers.push_back(std::move(buffer));

        std::cerr << "[Voice] Playing audio file: " << wavPath << std::endl;
        return true;
    }
}

// ------------------------------------------------------------
// [Voice] One-shot voice command
// ------------------------------------------------------------
CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);

    if (transcript.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_NO_SPEECH"),
            false,
            sf::Color::Red,
            "ERR_VOICE_NO_SPEECH",
            "No speech detected",
            "error"
        };
    }

    handleCommand(transcript);

    return {
        "> " + transcript,
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Voice command processed",
        "routine"
    };
}

// ------------------------------------------------------------
// [Voice] Continuous streaming mode
// ------------------------------------------------------------
CommandResult cmdVoiceStream([[maybe_unused]] const std::string& arg) {
    if (!Voice::g_state.ctx) {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_NO_CONTEXT"),
            false,
            sf::Color::Red,
            "ERR_VOICE_NO_CONTEXT",
            "Voice context missing",
            "error"
        };
    }

    if (VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, g_nlp)) {
        return {
            "[Voice] Streaming started.",
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Voice streaming started",
            "routine"
        };
    } else {
        return {
            ErrorManager::getUserMessage("ERR_VOICE_STREAM_FAIL"),
            false,
            sf::Color::Red,
            "ERR_VOICE_STREAM_FAIL",
            "Voice streaming failed",
            "error"
        };
    }
}

// ------------------------------------------------------------
// [Voice] Local TTS test (Microsoft David)
// ------------------------------------------------------------
CommandResult cmd_testTTS([[maybe_unused]] const std::string& arg) {
    CommandResult result;
    result.success = false;

    std::string text = arg.empty() ? "This is a Coqui voice test." : arg;

    std::cerr << "[Voice][Test] ===== BEGIN Coqui TTS TEST =====" << std::endl;
    std::cerr << "[Voice][Test] Text   : \"" << text << "\"" << std::endl;

    // 🔹 Ask Coqui to synthesize → get output path
    std::string wavPath = Voice::coquiSpeak(text, "p225", 1.0);
    if (wavPath.empty()) {
        result.message = "[Voice][Test] ERROR: Coqui TTS failed.";
        result.color   = sf::Color::Red;
        std::cerr << "[Voice][Test] ERROR: coquiSpeak returned empty path\n";
        return result;
    }

    std::cerr << "[Voice][Test] File   : " << wavPath << std::endl;

    // 🔹 Play the generated file
    Voice::playAudio(wavPath);

    result.success = true;
    result.message = "[Voice][Test] Coqui TTS playback requested.";
    result.color   = sf::Color::Green;
    return result;
}

// ------------------------------------------------------------
// [Voice] List installed SAPI voices
// ------------------------------------------------------------
CommandResult cmd_listVoices([[maybe_unused]] const std::string& arg) {
    auto cfg = aiConfig["voice"];
    std::ostringstream oss;

    std::string engine = cfg.value("engine", "sapi");

    if (engine == "coqui") {
        oss << "[Voice] Current Coqui TTS configuration:\n";
        oss << " - Model: " << "tts_models/en/ljspeech/vits" << "\n"; // must match your tts_bridge.py
        oss << " - Speaker: " << cfg.value("speaker", "default") << "\n";
        oss << " - Speed: " << cfg.value("speed", 1.0) << "\n";

        return {
            oss.str(),
            true,
            sf::Color::Yellow,
            "ERR_NONE",
            "Coqui voices listed",
            "debug"
        };
    }

#if defined(_WIN32)
    // Fallback: list SAPI voices
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        return { "[Voice][Error] Failed to initialize COM.", false,
                 sf::Color::Red, "ERR_TTS_COM", "COM init failed", "debug" };
    }

    IEnumSpObjectTokens* pEnum = nullptr;
    ULONG ulCount = 0;

    hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &pEnum);
    if (SUCCEEDED(hr) && pEnum) {
        pEnum->GetCount(&ulCount);
        oss << "[Voice] Found " << ulCount << " installed SAPI voices:\n";

        for (ULONG i = 0; i < ulCount; i++) {
            ISpObjectToken* pToken = nullptr;
            if (SUCCEEDED(pEnum->Next(1, &pToken, NULL)) && pToken) {
                WCHAR* pszDesc = nullptr;
                if (SUCCEEDED(SpGetDescription(pToken, &pszDesc)) && pszDesc) {
                    char buffer[512];
                    size_t converted = 0;
                    wcstombs_s(&converted, buffer, sizeof(buffer), pszDesc, _TRUNCATE);
                    oss << " - " << buffer << "\n";
                    ::CoTaskMemFree(pszDesc);
                }
                pToken->Release();
            }
        }

        pEnum->Release();
        CoUninitialize();

        return {
            oss.str(),
            true,
            sf::Color::Yellow,
            "ERR_NONE",
            "SAPI voices listed",
            "debug"
        };
    }

    if (pEnum) pEnum->Release();
    CoUninitialize();
    return {
        "[Voice][Error] Failed to enumerate SAPI voices.",
        false,
        sf::Color::Red,
        "ERR_TTS_ENUM",
        "Failed to list SAPI voices",
        "debug"
    };
#else
    return {
        "[Voice][Error] Voice listing is only supported on Windows (for SAPI).",
        false,
        sf::Color::Red,
        "ERR_UNSUPPORTED_PLATFORM",
        "Voice listing unsupported",
        "debug"
    };
#endif
}

// ------------------------------------------------------------
// [Debug] Speak a test line directly through SAPI
// ------------------------------------------------------------
CommandResult cmd_testSAPI([[maybe_unused]] const std::string& arg) {
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile("resources/test.wav")) {
        return {
            "[Audio] Failed to load resources/test.wav",
            false,
            sf::Color::Red,
            "ERR_AUDIO_LOAD",
            "Audio load failed",
            "error"
        };
    }

    sf::Sound sound(buffer);
    sound.play();

    std::cerr << "[Audio] Playing test.wav..." << std::endl;

    while (sound.getStatus() == sf::Sound::Status::Playing) {
        sf::sleep(sf::milliseconds(100));
    }

    return {
        "[Audio] Test file played successfully.",
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Audio playback succeeded",
        "routine"
    };
}

// ------------------------------------------------------------
// [Voice] Get current SAPI output device
// ------------------------------------------------------------
CommandResult cmd_ttsDevice([[maybe_unused]] const std::string& arg) {
#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        return { "[Voice][Error] Failed to initialize COM.", false,
                 sf::Color::Red, "ERR_TTS_COM", "COM init failed", "debug" };
    }

    ISpVoice* pVoice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL,
                          IID_ISpVoice, (void**)&pVoice);

    if (FAILED(hr) || !pVoice) {
        CoUninitialize();
        return { "[Voice][Error] Failed to create SAPI voice instance.", false,
                 sf::Color::Red, "ERR_TTS_INIT", "SAPI init failed", "debug" };
    }

    ISpObjectToken* pAudioOut = nullptr;
    hr = pVoice->GetOutputObjectToken(&pAudioOut);

    std::ostringstream oss;
    if (SUCCEEDED(hr) && pAudioOut) {
        WCHAR* pszDesc = nullptr;
        if (SUCCEEDED(SpGetDescription(pAudioOut, &pszDesc)) && pszDesc) {
            char buffer[512];
            size_t converted = 0;
            wcstombs_s(&converted, buffer, sizeof(buffer), pszDesc, _TRUNCATE);
            oss << "[Voice] Current SAPI output device: " << buffer << "\n";
            ::CoTaskMemFree(pszDesc);
        }
        pAudioOut->Release();
    } else {
        oss << "[Voice] Could not retrieve current output device.\n";
    }

    pVoice->Release();
    CoUninitialize();

    return { oss.str(), true, sf::Color::Yellow,
             "ERR_NONE", "Device info", "debug" };
#else
    return { "[Voice][Error] Device query only works on Windows.", false,
             sf::Color::Red, "ERR_UNSUPPORTED_PLATFORM", "Device query unsupported", "debug" };
#endif
}
