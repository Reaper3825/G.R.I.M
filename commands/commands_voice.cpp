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
#include "voice/voice.hpp"
#include "voice/voice_stream.hpp"
#include "commands_core.hpp"
#include "voice/voice_speak.hpp"
#include "resources.hpp"
#include "nlp/nlp.hpp"
#include "logger.hpp"
#include "core/audio_core.hpp"



// ---------------------------------------------------------
// Standard headers
// ---------------------------------------------------------
#include <iostream>
#include <sstream>
#include <memory>
#include <vector>


// ====================================================
// [Voice] One-shot voice command
// ====================================================
CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);

    if (transcript.empty()) {
        return {
            false,                                             // success
            ErrorManager::getUserMessage("ERR_VOICE_NO_SPEECH"), // message
            "ERR_VOICE_NO_SPEECH",                             // errorCode
            "error",                                           // category
            "No speech detected",                              // voice
            Colors::Red                                        // color
        };
    }

    LOG_DEBUG("Voice", "Received transcript: " + transcript);
    // NOTE: handleCommand() is called by the wake key system in wake_key.cpp
    // We just return the transcript here without executing it again

    return {
        true,                          // success
        "> " + transcript,             // message
        "ERR_NONE",                    // errorCode
        "routine",                     // category
        "",                            // voice (empty to prevent speaking)
        Colors::Cyan                   // color
    };
}

// ====================================================
// [Voice] Continuous streaming mode
// ====================================================
CommandResult cmdVoiceStream([[maybe_unused]] const std::string& arg) {
    if (!Voice::g_state.ctx) {
        LOG_ERROR("Voice", "No context available for streaming");
        return {
            false,                                                 // success
            ErrorManager::getUserMessage("ERR_VOICE_NO_CONTEXT"),  // message
            "ERR_VOICE_NO_CONTEXT",                                // errorCode
            "error",                                               // category
            "Voice context missing",                               // voice
            Colors::Red                                            // color
        };
    }

    if (VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, g_nlp)) {
        LOG_DEBUG("Voice", "Voice streaming started");
        return {
            true,                               // success
            "[Voice] Streaming started.",       // message
            "ERR_NONE",                         // errorCode
            "routine",                          // category
            "Voice streaming started",          // voice
            Colors::Green                       // color
        };
    } else {
        LOG_ERROR("Voice", "Voice streaming failed");
        return {
            false,                                                   // success
            ErrorManager::getUserMessage("ERR_VOICE_STREAM_FAIL"),   // message
            "ERR_VOICE_STREAM_FAIL",                                 // errorCode
            "error",                                                 // category
            "Voice streaming failed",                                // voice
            Colors::Red                                              // color
        };
    }
}

// ====================================================
// [Voice] Local TTS test (Microsoft David / Coqui)
// ====================================================
CommandResult cmd_testTTS([[maybe_unused]] const std::string& arg) {
    CommandResult result;
    result.success = false;

    std::string text = arg.empty() ? "This is a Coqui voice test." : arg;

    LOG_DEBUG("Voice", "===== BEGIN Coqui TTS TEST =====");
    LOG_DEBUG("Voice", "Text: \"" + text + "\"");

    // 🔹 Ask Coqui to synthesize → get output path
    std::string wavPath = Voice::coquiSpeak(text, "p225", 1.0);
    if (wavPath.empty()) {
        LOG_ERROR("Voice", "Coqui TTS failed (empty output path)");
        result.message = "[Voice][Test] ERROR: Coqui TTS failed.";
        result.color   = Colors::Red;
        return result;
    }

    LOG_DEBUG("Voice", "Generated WAV file: " + wavPath);

    // 🔹 Play the generated file
    Voice::playAudio(wavPath);

    result.success = true;
    result.message = "[Voice][Test] Coqui TTS playback requested.";
    result.color   = Colors::Green;
    return result;
}

// ====================================================
// [Voice] List installed SAPI voices
// ====================================================
CommandResult cmd_listVoices([[maybe_unused]] const std::string& arg) {
    auto cfg = aiConfig["voice"];
    std::ostringstream oss;

    std::string engine = cfg.value("engine", "sapi");

    if (engine == "coqui") {
        oss << "[Voice] Current Coqui TTS configuration:\n";
        oss << " - Model: " << "tts_models/en/ljspeech/vits" << "\n";
        oss << " - Speaker: " << cfg.value("speaker", "default") << "\n";
        oss << " - Speed: " << cfg.value("speed", 1.0) << "\n";

        LOG_DEBUG("Voice", "Listing Coqui configuration");
        return {
            true,                           // success
            oss.str(),                      // message
            "ERR_NONE",                     // errorCode
            "debug",                        // category
            "Coqui voices listed",          // voice
            Colors::Yellow                  // color
        };
    }

#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        LOG_ERROR("Voice", "Failed to initialize COM for SAPI voice listing");
        return {
            false,                                           // success
            "[Voice][Error] Failed to initialize COM.",      // message
            "ERR_TTS_COM",                                   // errorCode
            "debug",                                         // category
            "COM init failed",                               // voice
            Colors::Red                                      // color
        };
    }

    IEnumSpObjectTokens* pEnum = nullptr;
    ULONG ulCount = 0;

    hr = SpEnumTokens(SPCAT_VOICES, NULL, NULL, &pEnum);
    if (SUCCEEDED(hr) && pEnum) {
        pEnum->GetCount(&ulCount);
        oss << "[Voice] Found " << ulCount << " installed SAPI voices:\n";
        LOG_DEBUG("Voice", "Enumerating " + std::to_string(ulCount) + " SAPI voices");

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
            true,                       // success
            oss.str(),                  // message
            "ERR_NONE",                 // errorCode
            "debug",                    // category
            "SAPI voices listed",       // voice
            Colors::Yellow              // color
        };
    }

    if (pEnum) pEnum->Release();
    CoUninitialize();
    LOG_ERROR("Voice", "Failed to enumerate SAPI voices");
    return {
        false,                                                      // success
        "[Voice][Error] Failed to enumerate SAPI voices.",         // message
        "ERR_TTS_ENUM",                                             // errorCode
        "debug",                                                    // category
        "Failed to list SAPI voices",                              // voice
        Colors::Red                                                 // color
    };
#else
    LOG_ERROR("Voice", "Voice listing unsupported on non-Windows platforms");
    return {
        false,                                                                            // success
        "[Voice][Error] Voice listing is only supported on Windows (for SAPI).",         // message
        "ERR_UNSUPPORTED_PLATFORM",                                                       // errorCode
        "debug",                                                                          // category
        "Voice listing unsupported",                                                      // voice
        Colors::Red                                                                       // color
    };
#endif
}

// ====================================================
// [Debug] Speak a test line directly through SAPI
// ====================================================
CommandResult cmd_testSAPI([[maybe_unused]] const std::string& arg) {
    std::string path = "resources/test.wav";
    LOG_DEBUG("Audio", "Testing playback of: " + path);

    if (!std::filesystem::exists(path)) {
        LOG_ERROR("Audio", "File not found: " + path);
        return {
            false,                           // success
            "[Audio] Test file not found.",  // message
            "ERR_AUDIO_MISSING",             // errorCode
            "error",                         // category
            "Missing audio file",            // voice
            Colors::Red                      // color
        };
    }

    if (!Audio::playWav(path)) {
        LOG_ERROR("Audio", "Playback failed via AudioCore.");
        return {
            false,                           // success
            "[Audio] Playback failed.",      // message
            "ERR_AUDIO_FAIL",                // errorCode
            "error",                         // category
            "Audio playback failed",         // voice
            Colors::Red                      // color
        };
    }

    return {
        true,                                           // success
        "[Audio] Test file played successfully.",       // message
        "ERR_NONE",                                     // errorCode
        "routine",                                      // category
        "Audio playback succeeded",                     // voice
        Colors::Green                                   // color
    };
}


// ====================================================
// [Voice] Get current SAPI output device
// ====================================================
CommandResult cmd_ttsDevice([[maybe_unused]] const std::string& arg) {
#if defined(_WIN32)
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        LOG_ERROR("Voice", "Failed to initialize COM for device query");
        return {
            false,                                           // success
            "[Voice][Error] Failed to initialize COM.",      // message
            "ERR_TTS_COM",                                   // errorCode
            "debug",                                         // category
            "COM init failed",                               // voice
            Colors::Red                                      // color
        };
    }

    ISpVoice* pVoice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL,
                          IID_ISpVoice, (void**)&pVoice);

    if (FAILED(hr) || !pVoice) {
        CoUninitialize();
        LOG_ERROR("Voice", "Failed to create SAPI voice instance");
        return {
            false,                                                      // success
            "[Voice][Error] Failed to create SAPI voice instance.",    // message
            "ERR_TTS_INIT",                                             // errorCode
            "debug",                                                    // category
            "SAPI init failed",                                         // voice
            Colors::Red                                                 // color
        };
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
            LOG_DEBUG("Voice", std::string("Current output device: ") + buffer);
        }
        pAudioOut->Release();
    } else {
        oss << "[Voice] Could not retrieve current output device.\n";
        LOG_ERROR("Voice", "Could not retrieve current output device");
    }

    pVoice->Release();
    CoUninitialize();

    return {
        true,               // success
        oss.str(),          // message
        "ERR_NONE",         // errorCode
        "debug",            // category
        "Device info",      // voice
        Colors::Yellow      // color
    };
#else
    LOG_ERROR("Voice", "Device query unsupported on this platform");
    return {
        false,                                                      // success
        "[Voice][Error] Device query only works on Windows.",      // message
        "ERR_UNSUPPORTED_PLATFORM",                                 // errorCode
        "debug",                                                    // category
        "Device query unsupported",                                 // voice
        Colors::Red                                                 // color
    };
#endif
}

// ====================================================
// Nevermind Command
// ====================================================
CommandResult cmdNevermind(const std::string& arg)
{
    (void)arg;
    g_lastIntent = {}; // reset last NLP intent
    LOG_DEBUG("Command", "User cancelled last command with 'nevermind'");

    // Optionally stop active speech playback
    if (Voice::isSpeaking()) {
    LOG_DEBUG("Command", "Would stop active TTS playback — placeholder (AudioCore).");
    // TODO: integrate mixer stop logic once audio_core supports it.
    }


    return {
        true,                          // success
        "Alright, cancelled.",         // message
        "ERR_NONE",                    // errorCode
        "routine",                     // category
        "Alright, cancelled.",         // voice
        Color(128, 128, 255)           // color
    };
}
