#include "host_voice.hpp"
#include "plugins/plugin_logging.hpp"
#include <iostream>
#include <sstream>
#include <vector>

// ====================================================
// [Voice] One-shot voice command
// ====================================================
CommandResult cmdVoice([[maybe_unused]] const std::string& arg) {
    std::string transcript = hostVoice::captureAndTranscribeSpeech();

    if (transcript.empty()) {
        return {
            false,                                             // success
            hostVoice::errorMessage("ERR_VOICE_NO_SPEECH"), // message
            "ERR_VOICE_NO_SPEECH",                             // errorCode
            "error",                                           // category
            "No speech detected",                              // voice
            Colors::Red                                        // color
        };
    }

    LOG_DEBUG("Voice", "Received transcript: " + transcript);
    // Return the transcript to the caller; wake-key captures use ai_process()
    // directly and therefore do not execute this command a second time.

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
    if (!hostVoice::hasContext()) {
        LOG_ERROR("Voice", "No context available for streaming");
        return {
            false,                                                 // success
            hostVoice::errorMessage("ERR_VOICE_NO_CONTEXT"),  // message
            "ERR_VOICE_NO_CONTEXT",                                // errorCode
            "error",                                               // category
            "Voice context missing",                               // voice
            Colors::Red                                            // color
        };
    }

    if (hostVoice::startStream()) {
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
            hostVoice::errorMessage("ERR_VOICE_STREAM_FAIL"),   // message
            "ERR_VOICE_STREAM_FAIL",                                 // errorCode
            "error",                                                 // category
            "Voice streaming failed",                                // voice
            Colors::Red                                              // color
        };
    }
}

// ====================================================
// [Voice] Local TTS test (XTTS v2)
// ====================================================
CommandResult cmd_testTTS([[maybe_unused]] const std::string& arg) {
    CommandResult result;
    result.success = false;

    std::string text = arg.empty() ? "This is a Coqui XTTS version two voice test." : arg;

    LOG_DEBUG("Voice", "===== BEGIN Coqui XTTS v2 TEST =====");
    LOG_DEBUG("Voice", "Text: \"" + text + "\"");
    
    // Check if XTTS v2 is enabled
    if (hostVoice::isXTTSv2Enabled()) {
        LOG_DEBUG("Voice", "Using XTTS v2 model with language: " + hostVoice::getLanguage());
    }

    // 🔹 Ask Coqui XTTS v2 to synthesize → get output path
    std::string wavPath = hostVoice::coquiSpeak(text, "default", 1.0);
    if (wavPath.empty()) {
        LOG_ERROR("Voice", "Coqui XTTS v2 TTS failed (empty output path)");
        result.message = "[Voice][Test] ERROR: Coqui XTTS v2 TTS failed.";
        result.color   = Colors::Red;
        return result;
    }

    LOG_DEBUG("Voice", "Generated WAV file: " + wavPath);

    // 🔹 Play the generated file
    hostVoice::playAudio(wavPath);

    result.success = true;
    result.message = "[Voice][Test] Coqui XTTS v2 playback requested.";
    result.color   = Colors::Green;
    return result;
}

// ====================================================
// [Voice] List installed SAPI voices
// ====================================================
CommandResult cmd_listVoices([[maybe_unused]] const std::string& arg) {
    auto cfg = hostVoice::config();
    std::ostringstream oss;

    std::string engine = cfg.value("engine", "sapi");

    if (engine == "coqui") {
        oss << "[Voice] Current Coqui XTTS v2 configuration:\n";
        
        if (hostVoice::isXTTSv2Enabled()) {
            oss << " - Model: tts_models/multilingual/multi-dataset/xtts_v2\n";
            oss << " - Engine: XTTS v2 (Voice Cloning)\n";
            oss << " - Language: " << hostVoice::getLanguage() << "\n";
            oss << " - Supported Languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi\n";
        } else {
            oss << " - Model: Legacy Coqui TTS (VITS)\n";
        }
        
        oss << " - Speaker: " << cfg.value("speaker", "default") << "\n";
        oss << " - Speed: " << cfg.value("speed", 1.0) << "\n";

        LOG_DEBUG("Voice", "Listing Coqui XTTS v2 configuration");
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
    const auto sapi = hostVoice::listSapiVoices();
    if (!sapi.at("com_ok").get<bool>())
        return {false, "[Voice][Error] Failed to initialize COM.", "ERR_TTS_COM", "debug", "COM init failed", Colors::Red};
    if (sapi.at("enumerated").get<bool>()) {
        oss << "[Voice] Found " << sapi.at("count").get<unsigned long>() << " installed SAPI voices:\n";
        for (const auto& name : sapi.at("voices")) oss << " - " << name.get<std::string>() << "\n";
        return {true, oss.str(), "ERR_NONE", "debug", "SAPI voices listed", Colors::Yellow};
    }
    return {false, "[Voice][Error] Failed to enumerate SAPI voices.", "ERR_TTS_ENUM", "debug", "Failed to list SAPI voices", Colors::Red};
#else
    return {false, "[Voice][Error] Voice listing is only supported on Windows (for SAPI).", "ERR_UNSUPPORTED_PLATFORM", "debug", "Voice listing unsupported", Colors::Red};
#endif
}
// ====================================================
// [Debug] Speak a test line directly through SAPI
// ====================================================
CommandResult cmd_testSAPI([[maybe_unused]] const std::string& arg) {
    std::string path = hostVoice::paths().at("test_wav").get<std::string>();
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

    if (!hostVoice::playWav(path)) {
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
    const auto device = hostVoice::sapiOutputDevice();
    if (!device.at("com_ok").get<bool>())
        return {false, "[Voice][Error] Failed to initialize COM.", "ERR_TTS_COM", "debug", "COM init failed", Colors::Red};
    if (!device.at("voice_created").get<bool>())
        return {false, "[Voice][Error] Failed to create SAPI voice instance.", "ERR_TTS_INIT", "debug", "SAPI init failed", Colors::Red};
    std::ostringstream oss;
    if (device.at("has_output").get<bool>()) {
        if (!device.at("description").is_null())
            oss << "[Voice] Current SAPI output device: " << device.at("description").get<std::string>() << "\n";
    } else {
        oss << "[Voice] Could not retrieve current output device.\n";
    }
    return {true, oss.str(), "ERR_NONE", "debug", "Device info", Colors::Yellow};
#else
    return {false, "[Voice][Error] Device query only works on Windows.", "ERR_UNSUPPORTED_PLATFORM", "debug", "Device query unsupported", Colors::Red};
#endif
}
// ====================================================
// Nevermind Command
// ====================================================
CommandResult cmdNevermind(const std::string& arg)
{
    (void)arg;
    LOG_DEBUG("Command", "User cancelled last command with 'nevermind'");

    // Optionally stop active speech playback
    if (hostVoice::isSpeaking()) {
        LOG_DEBUG("Command", "Would stop active TTS playback — placeholder (AudioCore).");
        // TODO: integrate mixer stop logic once audio_core supports it.
    }

    return {
        true,                          // success
        "Alright, cancelled.",         // message
        "ERR_NONE",                    // errorCode
        "cancellation",                // category - CHANGED from "routine" to "cancellation"
        "Alright, cancelled.",         // voice
        Color(128, 128, 255)           // color
    };
}

// ====================================================
// [Voice] Create speaker embedding (XTTS v2)
// ====================================================
CommandResult cmd_createEmbedding(const std::string& arg) {
    // Parse: speaker_id path/to/voice.wav
    std::istringstream iss(arg);
    std::string speaker_id, ref_path;
    
    iss >> speaker_id;
    std::getline(iss, ref_path);
    ref_path = ref_path.substr(ref_path.find_first_not_of(" \t"));  // Trim leading whitespace
    
    if (speaker_id.empty() || ref_path.empty()) {
        LOG_ERROR("Voice", "Invalid arguments for create_embedding");
        return {
            false,
            "[Voice] Usage: create_embedding <speaker_id> <audio_file_path>",
            "ERR_INVALID_ARGS",
            "error",
            "Invalid arguments",
            Colors::Red
        };
    }
    
    LOG_DEBUG("Voice", "Creating embedding for " + speaker_id + " from " + ref_path);
    
    // Send command to Coqui bridge
    nlohmann::json req = {
        {"command", "create_embedding"},
        {"speaker", speaker_id},
        {"reference_path", ref_path}
    };
    
    // This would need to be sent to the coqui_bridge via the pipe
    // For now, return a placeholder result
    // TODO: Integrate with voice_speak.cpp pipe communication
    
    return {
        true,
        "[Voice] Embedding creation queued for: " + speaker_id,
        "ERR_NONE",
        "routine",
        "Creating voice embedding",
        Colors::Green
    };
}

// ====================================================
// [Voice] List speaker embeddings (XTTS v2)
// ====================================================
CommandResult cmd_listEmbeddings([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("Voice", "Listing speaker embeddings");
    
    std::ostringstream oss;
    oss << "[Voice] Cached Speaker Embeddings:\n\n";
    
    // Check embedding directory
    std::string embeddingDir = hostVoice::paths().at("embeddings").get<std::string>();
    
    if (!std::filesystem::exists(embeddingDir)) {
        oss << "No embeddings found. Create one with: create_embedding <speaker_id> <audio_file>\n";
        return {
            true,
            oss.str(),
            "ERR_NONE",
            "debug",
            "No embeddings found",
            Colors::Yellow
        };
    }
    
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(embeddingDir)) {
        if (entry.path().extension() == ".npz") {
            auto speaker_id = entry.path().stem().string();
            auto size_kb = entry.file_size() / 1024;
            
            oss << " - " << speaker_id << " (" << size_kb << " KB)\n";
            count++;
        }
    }
    
    if (count == 0) {
        oss << "No embeddings found.\n";
    } else {
        oss << "\nTotal: " << count << " embeddings\n";
        oss << "\nUsage: Set speaker in ai_config.json to use cached embedding.\n";
    }
    
    return {
        true,
        oss.str(),
        "ERR_NONE",
        "debug",
        count > 0 ? "Embeddings listed" : "No embeddings found",
        Colors::Cyan
    };
}
