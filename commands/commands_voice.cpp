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




namespace {
    bool playWavFile(const std::string& wavPath) {
        static std::vector<sf::Sound> sounds; // keep sounds alive until finished
        sf::SoundBuffer* buffer = new sf::SoundBuffer();

        if (!buffer->loadFromFile(wavPath)) {
            std::cerr << "[Voice][Error] Failed to load audio: " << wavPath << std::endl;
            delete buffer;
            return false;
        }

        sf::Sound sound;
        sound.setBuffer(*buffer);
        sound.play();

        // push_back moves + copies the sf::Sound (but still needs buffer lifetime)
        sounds.push_back(sound);

        std::cout << "[Voice] Playing audio file: " << wavPath << std::endl;
        return true;
    }
}
// useful for debug logging

// Externals
extern nlohmann::json aiConfig;
extern nlohmann::json longTermMemory;
extern std::vector<Timer> timers;
extern NLP g_nlp;
extern std::string g_inputBuffer;
extern std::filesystem::path g_currentDir;

// ------------------------------------------------------------
// [Voice] One-shot voice command
// ------------------------------------------------------------
CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    // 🔹 Run Whisper transcription
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

    // 🔹 Inject transcript back into GRIM as if user typed it
    handleCommand(transcript);

    // 🔹 Show transcript in history (cyan) without re-speaking
    return {
        "> " + transcript,         // mimic console input style
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

    if (VoiceStream::start(Voice::g_state.ctx, nullptr, timers, longTermMemory, g_nlp)) {
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
CommandResult cmd_testTTS([[maybe_unused]] const std::string& arg) {
    std::string testLine = "This is GRIM testing with Microsoft David Desktop voice.";

    bool ok = Voice::speakLocal(testLine, "Microsoft David Desktop");

    if (ok) {
        return {
            "[Voice] Local TTS (David) spoken successfully.",
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Local TTS test line spoken",
            "debug"
        };
    } else {
        return {
            "[Voice][Error] Local TTS playback failed.",
            false,
            sf::Color::Red,
            "ERR_TTS_PLAYBACK",
            "Local TTS playback failed",
            "debug"
        };
    }
}
CommandResult cmd_listVoices([[maybe_unused]] const std::string& arg) {
#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        return { "[Voice][Error] Failed to initialize COM.", false,
                 sf::Color::Red, "ERR_TTS_COM", "COM init failed", "debug" };
    }

    IEnumSpObjectTokens* pEnum = nullptr;
    ISpObjectToken* pToken = nullptr;
    ULONG ulCount = 0;

    hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &pEnum);
    if (SUCCEEDED(hr) && pEnum) {
        pEnum->GetCount(&ulCount);
        std::ostringstream oss;
        oss << "[Voice] Found " << ulCount << " installed voices:\n";

        for (ULONG i = 0; i < ulCount; i++) {
            pToken = nullptr;
            if (SUCCEEDED(pEnum->Next(1, &pToken, NULL)) && pToken) {
                WCHAR* pszDesc = nullptr;
                if (SUCCEEDED(SpGetDescription(pToken, &pszDesc)) && pszDesc) {
                    char buffer[512];
                    wcstombs(buffer, pszDesc, sizeof(buffer));
                    oss << " - " << buffer << "\n";
                    ::CoTaskMemFree(pszDesc);
                }
                pToken->Release();
            }
        }

        pEnum->Release();
        CoUninitialize();

        return { oss.str(), true, sf::Color::Yellow,
                 "ERR_NONE", "Voices listed", "debug" };
    }

    if (pEnum) pEnum->Release();
    CoUninitialize();
    return { "[Voice][Error] Failed to enumerate voices.", false,
             sf::Color::Red, "ERR_TTS_ENUM", "Failed to list voices", "debug" };
#else
    return { "[Voice][Error] Voice listing is only supported on Windows.", false,
             sf::Color::Red, "ERR_UNSUPPORTED_PLATFORM", "Voice listing unsupported", "debug" };
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

    sf::Sound sound;
    sound.setBuffer(buffer);
    sound.play();

    std::cout << "[Audio] Playing test.wav..." << std::endl;

    // Block until finished (simple test loop)
    while (sound.getStatus() == sf::Sound::Playing) {
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

    // Get current audio output object token (✅ correct API)
    ISpObjectToken* pAudioOut = nullptr;
    hr = pVoice->GetOutputObjectToken(&pAudioOut);

    std::ostringstream oss;
    if (SUCCEEDED(hr) && pAudioOut) {
        WCHAR* pszDesc = nullptr;
        if (SUCCEEDED(SpGetDescription(pAudioOut, &pszDesc)) && pszDesc) {
            char buffer[512];
            wcstombs(buffer, pszDesc, sizeof(buffer));
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
