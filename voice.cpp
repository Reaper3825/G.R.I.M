#include "voice.hpp"
#include "resources.hpp"
#include "ai.hpp"
#include "voice_speak.hpp"
#include "system_detect.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"

#include <whisper.h>
#include <portaudio.h>
#include <filesystem>
#include <mutex>
#include <iostream>
#include <sstream>
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

    std::cerr << "[DEBUG][Voice] Looking for Whisper model at: " << modelPath << "\n";

    if (!fs::exists(modelPath)) {
        std::cerr << "[ERROR][Voice] Whisper model missing: " << modelPath << "\n";
        ErrorManager::report("ERR_VOICE_NOT_INITIALIZED");
        return false;
    }

    // Use default whisper context parameters
    whisper_context_params wparams = whisper_context_default_params();

    g_state.ctx = whisper_init_from_file_with_params(modelPath.string().c_str(), wparams);
    if (!g_state.ctx) {
        std::cerr << "[ERROR][Voice] Failed to load Whisper model: " << modelPath << "\n";
        ErrorManager::report("ERR_VOICE_TRANSCRIBE_FAIL");
        return false;
    }

    std::cout << "[Voice] Whisper model loaded: " << modelPath.filename().string() << "\n";
    return true;
}

// ============================================================
// Voice Input (Speech → Text)
// ============================================================
std::string runVoiceDemo(nlohmann::json& aiConfig, nlohmann::json& longTermMemory) {
    (void) longTermMemory;
    std::cerr << "[DEBUG][Voice] Entering runVoiceDemo()\n";

    // Load thresholds from config
    g_silenceThreshold = aiConfig["voice"].value("silence_threshold", 0.02f);
    g_silenceTimeoutMs = aiConfig["voice"].value("silence_timeout_ms", 4000);
    g_state.minSpeechMs  = aiConfig["whisper"].value("min_speech_ms", 500);
    g_state.minSilenceMs = aiConfig["whisper"].value("min_silence_ms", 1200);
    g_state.inputDeviceIndex = aiConfig["voice"].value("input_device_index", -1);

    // 🔹 Ensure Whisper model is loaded
    if (!ensureWhisperLoaded(aiConfig)) {
        return "";
    }

    if (Pa_Initialize() != paNoError) {
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

    AudioData data;
    PaStream* stream;
    int deviceIndex = (g_state.inputDeviceIndex >= 0) ? g_state.inputDeviceIndex : Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice || deviceIndex < 0 || deviceIndex >= Pa_GetDeviceCount()) {
        Pa_Terminate();
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    std::cerr << "[DEBUG][Voice] Using input device: " << devInfo->name << "\n";

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    if (Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 512, paNoFlag, recordCallback, &data) != paNoError) {
        Pa_Terminate();
        ErrorManager::report("ERR_VOICE_NO_CONTEXT");
        return "";
    }

    Pa_StartStream(stream);
    std::cerr << "[DEBUG][Voice] " << ResponseManager::get("voice_start") << "\n";

    std::vector<float> rollingBuffer;
    auto lastSpeech = std::chrono::steady_clock::now();
    auto speechStart = lastSpeech;
    bool inSpeech = false;

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
                        std::cerr << "[DEBUG][Voice] Speech started\n";
                    }
                    lastSpeech = std::chrono::steady_clock::now();
                    rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
                } else if (inSpeech) {
                    auto msSinceSpeech = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now() - lastSpeech).count();
                    auto msSpeech = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        lastSpeech - speechStart).count();

                    if (msSinceSpeech >= g_state.minSilenceMs && msSpeech >= g_state.minSpeechMs) {
                        std::cerr << "[DEBUG][Voice] End of speech detected\n";
                        break;
                    }
                    if (msSinceSpeech >= g_silenceTimeoutMs) {
                        std::cerr << "[DEBUG][Voice] Timeout reached\n";
                        break;
                    }
                }
            }
        }
        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cerr << "[DEBUG][Voice] Stream stopped\n";

    std::string transcript;
    if (!rollingBuffer.empty()) {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.no_timestamps = true;

        if (whisper_full(g_state.ctx, wparams, rollingBuffer.data(), rollingBuffer.size()) == 0) {
            int n = whisper_full_n_segments(g_state.ctx);
            for (int i = 0; i < n; i++) {
                transcript += whisper_full_get_segment_text(g_state.ctx, i);
                transcript += " ";
            }
        }
    }
    if (!transcript.empty() && transcript.back() == ' ')
        transcript.pop_back();

    if (!transcript.empty()) {
        std::cerr << "[DEBUG][Voice] " << ResponseManager::get("voice_heard")
                  << " \"" << transcript << "\"\n";
    } else {
        ErrorManager::report("ERR_VOICE_NO_SPEECH");
    }

    return transcript;
}

// ============================================================
// Shutdown
// ============================================================
void shutdown() {
    std::cerr << "[DEBUG][Voice] Shutdown called\n";
    if (g_state.ctx) {
        whisper_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
}

} // namespace Voice
