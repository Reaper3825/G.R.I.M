#include "voice.hpp"
#include "resources.hpp"
#include "ai/ai.hpp"
#include "voice_speak.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "logger.hpp" 
#include "popup_ui/popup_ui.hpp"  // ✅ Added
#include <whisper.h>
#include <portaudio.h>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdio>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <cmath>

namespace fs = std::filesystem;

namespace Voice {

// ---------------- State ----------------
State g_state;

static double g_silenceThreshold = 0.02;
static int g_silenceTimeoutMs = 4000;

// ---------------- Audio Data ----------------
struct AudioData {
    std::vector<float> buffer;
    std::mutex mtx;
    bool ready = false;
};

// ============================================================
// PortAudio Helpers
// ============================================================
static int recordCallback(const void* input,
                          void* /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo*,
                          PaStreamCallbackFlags,
                          void* userData) {
    AudioData* data = reinterpret_cast<AudioData*>(userData);
    const float* in = reinterpret_cast<const float*>(input);
    if (in) {
        std::lock_guard<std::mutex> lock(data->mtx);
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
    }
    return paContinue;
}

// ============================================================
// Silence Detection
// ============================================================
static bool isSilence(const std::vector<float>& pcm) {
    if (pcm.empty()) return true;
    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    double rms = std::sqrt(energy);
    
    // ✅ Log RMS for debugging
    static int logCounter = 0;
    if (++logCounter % 100 == 0) { // Log every 100th check to avoid spam
        LOG_TRACE("Voice", "RMS: " + std::to_string(rms) + " (threshold: " + std::to_string(g_silenceThreshold) + ")");
    }
    
    return rms < g_silenceThreshold;
}

// ============================================================
// Lazy Whisper Initialization
// ============================================================
static bool ensureWhisperLoaded(const nlohmann::json& aiConfig) {
    if (g_state.ctx) return true;

    // Pick model name from config, default to base English
    std::string modelName = "ggml-base.en.bin";
    if (aiConfig.contains("whisper") && aiConfig["whisper"].contains("whisper_model")) {
        modelName = aiConfig["whisper"].value("whisper_model", modelName);
    }

    // Resolve model path against resource root
    fs::path modelPath = fs::path(getResourcePath()) / "models" / modelName;

    LOG_DEBUG("Voice", "Looking for Whisper model at: " + modelPath.string());

    if (!fs::exists(modelPath)) {
        LOG_ERROR("Voice", "Whisper model missing: " + modelPath.string());
        ErrorManager::report("ERR_VOICE_NOT_INITIALIZED");
        return false;
    }


   whisper_context_params wparams = whisper_context_default_params();
    wparams.use_gpu = true;  // enable GPU acceleration
    LOG_DEBUG("Voice", std::string("Whisper GPU flag set: ") + (wparams.use_gpu ? "true" : "false"));

    std::string modelPathUtf8(reinterpret_cast<const char*>(modelPath.u8string().c_str()));
  // convert filesystem::path → UTF-8 
    g_state.ctx = whisper_init_from_file_with_params(modelPathUtf8.c_str(), wparams);



    if (!g_state.ctx) {
        LOG_ERROR("Voice", "Failed to load Whisper model: " + modelPath.string());
        ErrorManager::report("ERR_VOICE_TRANSCRIBE_FAIL");
        return false;
    }

    LOG_PHASE("Whisper model load", true);
    return true;
}

// ============================================================
// Voice Input (Speech → Text)
// ============================================================
std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory) {
    (void) longTermMemory;
    LOG_DEBUG("Voice", "Entering runVoiceDemo()");

    // Load thresholds from config
    g_silenceThreshold   = aiConfig["voice"].value("silence_threshold", 0.02f);
    g_silenceTimeoutMs   = aiConfig["voice"].value("silence_timeout_ms", 1200);
    g_state.minSpeechMs  = aiConfig["whisper"].value("min_speech_ms", 500);
    g_state.minSilenceMs = aiConfig["whisper"].value("min_silence_ms", 1200);
    g_state.inputDeviceIndex = aiConfig["voice"].value("input_device_index", -1);

    // 🔹 Ensure Whisper model is loaded
    if (!ensureWhisperLoaded(aiConfig)) {
        return "";
    }
    
    // Temporarily redirect stderr to silence PortAudio debug output
#ifdef _WIN32
    int old_stderr = _dup(2);
    FILE* nul_file = fopen("nul", "w");
    _dup2(_fileno(nul_file), 2);
#else
    int old_stderr = dup(2);
    FILE* nul_file = fopen("/dev/null", "w");
    dup2(fileno(nul_file), 2);
#endif

    if (Pa_Initialize() != paNoError) {
#ifdef _WIN32
        _dup2(old_stderr, 2);
        _close(old_stderr);
#else
        dup2(old_stderr, 2);
        close(old_stderr);
#endif
        fclose(nul_file);
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

#ifdef _WIN32
    _dup2(old_stderr, 2);
    _close(old_stderr);
#else
    dup2(old_stderr, 2);
    close(old_stderr);
#endif
    fclose(nul_file);

    AudioData data;
    PaStream* stream;
    int deviceIndex = (g_state.inputDeviceIndex >= 0)
                        ? g_state.inputDeviceIndex
                        : Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice || deviceIndex < 0 || deviceIndex >= Pa_GetDeviceCount()) {
        Pa_Terminate();
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    if (devInfo) {
        LOG_DEBUG("Voice", "Using input device: " + std::string(devInfo->name));
    }

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    if (Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 512,
                      paNoFlag, recordCallback, &data) != paNoError) {
        Pa_Terminate();
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

    Pa_StartStream(stream);
    LOG_DEBUG("Voice", "Listening...");
    
    // ✅ Show popup when listening starts
    notifyPopupActivity();

    std::vector<float> rollingBuffer;
    auto lastSpeech  = std::chrono::steady_clock::now();
    auto speechStart = lastSpeech;
    auto listeningStart = std::chrono::steady_clock::now(); // ✅ Track total listening time
    bool inSpeech = false;
    
    // ✅ Maximum listening time (prevents infinite listening)
    const int maxListeningMs = aiConfig["whisper"].value("max_listening_ms", 10000); // 10 seconds default

    while (true) {
        {
            std::lock_guard<std::mutex> lock(data.mtx);
            if (data.buffer.size() >= 8000) {
                std::vector<float> chunk(data.buffer.begin(), data.buffer.begin() + 8000);
                data.buffer.erase(data.buffer.begin(), data.buffer.begin() + 8000);

                bool silent = isSilence(chunk);
                if (!silent) {
                    if (!inSpeech) {
                        speechStart = std::chrono::steady_clock::now();
                        inSpeech = true;
                        LOG_DEBUG("Voice", "Speech started");
                        notifyPopupActivity();  // ✅ Keep popup alive during speech
                    }
                    lastSpeech = std::chrono::steady_clock::now();
                    rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
                } else if (inSpeech) {
                    auto msSinceSpeech = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now() - lastSpeech).count();
                    auto msSpeech = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        lastSpeech - speechStart).count();

                    if (msSinceSpeech >= g_state.minSilenceMs && msSpeech >= g_state.minSpeechMs) {
                        LOG_DEBUG("Voice", "End of speech detected");
                        break;
                    }
                    if (msSinceSpeech >= g_silenceTimeoutMs) {
                        LOG_DEBUG("Voice", "Timeout reached");
                        break;
                    }
                }
            }
        }
        
        // ✅ Check if we've been listening too long without any speech
        auto totalListeningTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - listeningStart).count();
        
        if (!inSpeech && totalListeningTime > maxListeningMs) {
            LOG_DEBUG("Voice", "Max listening timeout reached (" + std::to_string(maxListeningMs) + "ms) - no speech detected");
            break;
        }
        
        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    LOG_DEBUG("Voice", "Stream stopped");
    
    // ✅ Keep popup alive during transcription processing
    notifyPopupActivity();

    std::string transcript;
    if (!rollingBuffer.empty()) {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        
        // ═══════════════════════════════════════════════════════════
        // Load optimized Whisper parameters from config to reduce hallucinations
        // ═══════════════════════════════════════════════════════════
        wparams.no_timestamps = true;
        
        // CRITICAL: Force English language (check both new and legacy locations)
        std::string lang = "en";  // Default fallback
        if (aiConfig["whisper"].contains("language")) {
            lang = aiConfig["whisper"]["language"].get<std::string>();
        } else if (aiConfig.contains("whisper_language")) {
            lang = aiConfig["whisper_language"].get<std::string>();  // Legacy location
        }
        wparams.language = lang.c_str();
        LOG_DEBUG("Voice", "Forcing language: " + lang);
        
        // Temperature: 0.0 = more deterministic (reduces hallucinations)
        wparams.temperature = aiConfig["whisper"].value("temperature", 0.0);
        
        // ✅ CRITICAL: Enable temperature fallback to catch hallucinations
        wparams.temperature_inc = 0.0;  // Don't increase temperature on retries
        
        // Max length: prefer shorter outputs
        wparams.max_len = aiConfig["whisper"].value("max_len", 1);
        
        // Beam size: more careful decoding
        wparams.beam_search.beam_size = aiConfig["whisper"].value("beam_size", 5);
        
        // Suppress blank outputs
        wparams.suppress_blank = aiConfig["whisper"].value("suppress_blank", true);
        wparams.token_timestamps = false;
        
        // No speech threshold: filter out noise
        wparams.no_speech_thold = aiConfig["whisper"].value("no_speech_threshold", 0.6);
        
        // ✅ NEW: Entropy threshold to detect gibberish/hallucinations
        wparams.entropy_thold = 2.4f;  // Reject high-entropy (random) outputs
        
        // ✅ NEW: Logprob threshold to reject low-confidence transcriptions
        wparams.logprob_thold = -1.0f;  // Reject outputs with low log probability
        
        // Initial prompt to guide the model toward command-style output
        std::string prompt = aiConfig["whisper"].value("initial_prompt", 
            "Voice commands: open notepad, close window, show time");
        wparams.initial_prompt = prompt.c_str();

        LOG_DEBUG("Voice", "Whisper params: temp=" + std::to_string(wparams.temperature) + 
                  " beam=" + std::to_string(wparams.beam_search.beam_size) +
                  " no_speech_thold=" + std::to_string(wparams.no_speech_thold) +
                  " entropy_thold=" + std::to_string(wparams.entropy_thold) +
                  " logprob_thold=" + std::to_string(wparams.logprob_thold) +
                  " lang=" + lang);

        if (whisper_full(g_state.ctx, wparams, rollingBuffer.data(),
                         rollingBuffer.size()) == 0) {
            int n = whisper_full_n_segments(g_state.ctx);
            for (int i = 0; i < n; i++) {
                transcript += whisper_full_get_segment_text(g_state.ctx, i);
                transcript += " ";
            }
        }
    }
    
    // Trim trailing space
    if (!transcript.empty() && transcript.back() == ' ')
        transcript.pop_back();

    // ✅ NEW: Post-process to filter hallucinations
    if (!transcript.empty()) {
        // Common Whisper hallucination patterns to filter
        const std::vector<std::string> hallucinationPatterns = {
            "blank_audio",
            "(blank audio)",
            "(clicking)",
            "(silence)",
            "(no audio)",
            "thank you for watching",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "state 0-1",  // Specific hallucination from logs
            "state zero one",
            "state 01",
            "[Music]",
            "[Applause]",
            "(music)",
            "(applause)",
            "♪",
            "♫"
        };
        
        std::string lowerTranscript = transcript;
        std::transform(lowerTranscript.begin(), lowerTranscript.end(), 
                      lowerTranscript.begin(), ::tolower);
        
        // Remove leading/trailing whitespace and punctuation
        lowerTranscript.erase(0, lowerTranscript.find_first_not_of(" \t\n\r.,!?"));
        lowerTranscript.erase(lowerTranscript.find_last_not_of(" \t\n\r.,!?") + 1);
        
        for (const auto& pattern : hallucinationPatterns) {
            if (lowerTranscript.find(pattern) != std::string::npos) {
                LOG_DEBUG("Voice", "Filtered hallucination pattern: \"" + pattern + "\" from: \"" + transcript + "\"");
                transcript.clear();
                break;
            }
        }
        
        // ✅ Additional check: reject very short gibberish (single letters/numbers)
        if (transcript.length() <= 2 && !std::isalpha(transcript[0])) {
            LOG_DEBUG("Voice", "Filtered short gibberish: \"" + transcript + "\"");
            transcript.clear();
        }
        
        // ✅ Check for repeated characters (another hallucination sign)
        if (transcript.length() > 3) {
            bool allSame = true;
            char first = std::tolower(transcript[0]);
            for (char c : transcript) {
                if (std::tolower(c) != first && !std::isspace(c)) {
                    allSame = false;
                    break;
                }
            }
            if (allSame) {
                LOG_DEBUG("Voice", "Filtered repeated character hallucination: \"" + transcript + "\"");
                transcript.clear();
            }
        }
    }

    if (!transcript.empty()) {
        LOG_DEBUG("Voice", "Heard speech: \"" + transcript + "\"");
    } else {
        LOG_DEBUG("Voice", "No speech detected or recognized");
        // Don't report error - silence is a valid state
    }

    return transcript;
}
void setWhisperContext(whisper_context* ctx) {
    g_state.ctx = ctx;
}

// ============================================================
// Shutdown
// ============================================================
void shutdown() {
    LOG_DEBUG("Voice", "Shutdown called");
    if (g_state.ctx) {
        whisper_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
}

// ============================================================
// Accessor
// ============================================================
whisper_context* getWhisperContext() {
    return g_state.ctx;
}

std::string Voice::detectWakeWord() {
    // In future: stream mic audio → run STT or keyword spotter
    // For now: simulate by polling a file or console input
    return ""; // Return "grim" when detected
}

} // namespace Voice
