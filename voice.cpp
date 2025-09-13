#include "voice.hpp"
#include "resources.hpp"
#include "whisper.h"

#include <iostream>
#include <vector>
#include <filesystem>
#include <portaudio.h>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------
#define SAMPLE_RATE        16000
#define FRAMES_PER_BUFFER   512
#define CHUNK_SIZE         (SAMPLE_RATE / 2) // 0.5 seconds

// ------------------------------------------------------------
// Global Whisper state
// ------------------------------------------------------------
struct whisper_context* g_whisperCtx = nullptr;
float g_lastSegmentConfidence = -1.0f;

// Config values (defined in ai.cpp, except min speech/silence)
extern nlohmann::json aiConfig;
extern double g_silenceThreshold;
extern int g_silenceTimeoutMs;
int g_minSpeechMs  = 500;  // Step 5: smarter silence detection
int g_minSilenceMs = 800;

// Environment baseline
static double g_envLevel = 0.0;

// ------------------------------------------------------------
// Audio Recording
// ------------------------------------------------------------
struct AudioData {
    std::vector<float> buffer;
    bool ready = false;
};

static int recordCallback(const void* input,
                          void* /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void* userData) {
    AudioData* data = reinterpret_cast<AudioData*>(userData);
    const float* in = reinterpret_cast<const float*>(input);

    if (in) {
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
    }

    return paContinue; // keep streaming
}

// ------------------------------------------------------------
// Smarter Silence Detection
// ------------------------------------------------------------
static bool isSilence(const std::vector<float>& pcm) {
    if (pcm.empty()) return true;

    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    double rms = std::sqrt(energy);

    double adjusted = std::max(0.0, rms - g_envLevel);

    std::cout << "[DEBUG][Voice] RMS=" << rms
              << " baseline=" << g_envLevel
              << " adjusted=" << adjusted
              << " threshold=" << g_silenceThreshold << "\n";

    return adjusted < g_silenceThreshold;
}

// ------------------------------------------------------------
// Whisper Initialization
// ------------------------------------------------------------
bool initWhisper(const std::string& modelName) {
    namespace fs = std::filesystem;

    fs::path modelPathFS =
        fs::path(getResourcePath()) / "models" / "whisper" / ("ggml-" + modelName + ".bin");

    std::cout << "[Voice] Attempting to load Whisper model: " << modelPathFS << "\n";

    if (!fs::exists(modelPathFS)) {
        std::cerr << "[Voice] ERROR: Model not found at " << modelPathFS << "\n";
        return false;
    }

    struct whisper_context_params wparams = whisper_context_default_params();
    g_whisperCtx = whisper_init_from_file_with_params(modelPathFS.string().c_str(), wparams);

    if (!g_whisperCtx) {
        std::cerr << "[Voice] ERROR: Failed to load Whisper model\n";
        return false;
    }

    // 🔹 Safe config loading
    g_silenceThreshold = aiConfig.value("silence_threshold", 0.00005);
    g_silenceTimeoutMs = aiConfig.value("silence_timeout_ms", 2500);

    if (aiConfig.contains("whisper") && aiConfig["whisper"].is_object()) {
        auto& w = aiConfig["whisper"];
        g_minSpeechMs  = w.value("min_speech_ms", 500);
        g_minSilenceMs = w.value("min_silence_ms", 800);
    } else {
        g_minSpeechMs  = 500;
        g_minSilenceMs = 800;
    }

    std::cout << "[Voice] Whisper initialized. "
              << "silence_threshold=" << g_silenceThreshold
              << ", silence_timeout=" << g_silenceTimeoutMs
              << ", min_speech_ms=" << g_minSpeechMs
              << ", min_silence_ms=" << g_minSilenceMs << "\n";

    return true;
}

bool initWhisper() {
    return initWhisper("small");
}

// ------------------------------------------------------------
// Voice Demo
// ------------------------------------------------------------
std::string runVoiceDemo(const std::string& /*modelPath*/,
                         nlohmann::json& longTermMemory) {
    if (!g_whisperCtx) {
        std::cerr << "[Voice] Whisper not initialized! Call initWhisper() first.\n";
        return "";
    }

    // Cached baseline
    if (longTermMemory.contains("voice_baseline")) {
        g_envLevel = longTermMemory["voice_baseline"].get<double>();
        std::cout << "[Voice] Loaded cached environment baseline: " << g_envLevel << "\n";
    }

    if (Pa_Initialize() != paNoError) {
        return "[ERROR] Failed to init PortAudio";
    }

    AudioData data;
    PaStream* stream;
    if (Pa_OpenDefaultStream(&stream,
                             1, 0,
                             paFloat32,
                             SAMPLE_RATE,
                             FRAMES_PER_BUFFER,
                             recordCallback,
                             &data) != paNoError) {
        Pa_Terminate();
        return "[ERROR] Failed to open audio stream";
    }

    Pa_StartStream(stream);
    std::cout << "[Voice] Listening...\n";

    std::vector<float> rollingBuffer;
    std::string transcript;
    int lastPrintedSegment = 0;

    auto lastSpeech = std::chrono::steady_clock::now();
    auto speechStart = std::chrono::steady_clock::now();
    bool inSpeech = false;

    while (true) {
        if (data.buffer.size() >= CHUNK_SIZE) {
            std::vector<float> chunk(data.buffer.begin(), data.buffer.begin() + CHUNK_SIZE);
            data.buffer.erase(data.buffer.begin(), data.buffer.begin() + CHUNK_SIZE);

            if (!isSilence(chunk)) {
                if (!inSpeech) {
                    speechStart = std::chrono::steady_clock::now();
                    inSpeech = true;
                }
                lastSpeech = std::chrono::steady_clock::now();
                rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
            } else {
                if (inSpeech) {
                    auto msSinceSpeech =
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - lastSpeech).count();
                    if (msSinceSpeech > g_minSilenceMs) {
                        std::cout << "[DEBUG][Voice] End of speech detected after "
                                  << msSinceSpeech << " ms silence\n";
                        break;
                    }
                }
            }

            if (rollingBuffer.size() > SAMPLE_RATE * 10) {
                rollingBuffer.erase(rollingBuffer.begin(),
                                    rollingBuffer.begin() + (rollingBuffer.size() - SAMPLE_RATE * 10));
                lastPrintedSegment = 0;
            }

            if (rollingBuffer.size() >= SAMPLE_RATE * 2) {
                // Normalize audio
                float maxVal = 0.0f;
                for (float s : rollingBuffer) maxVal = std::max(maxVal, std::fabs(s));
                if (maxVal > 0.0f) for (float& s : rollingBuffer) s /= maxVal;

                // 🔹 Safe Whisper params loading
                std::string strategy = "greedy";
                float temp = 0.2f;
                std::string lang = "en";

                if (aiConfig.contains("whisper") && aiConfig["whisper"].is_object()) {
                    auto& w = aiConfig["whisper"];
                    strategy = w.value("sampling_strategy", "greedy");
                    temp     = w.value("temperature", 0.2f);
                }
                lang = aiConfig.value("whisper_language", "en");

                whisper_full_params params =
                    whisper_full_default_params(
                        strategy == "beam" ? WHISPER_SAMPLING_BEAM_SEARCH
                                           : WHISPER_SAMPLING_GREEDY);

                params.language       = lang.empty() ? nullptr : lang.c_str();
                params.no_context     = false;
                params.single_segment = false;
                params.print_realtime = false;
                params.print_progress = false;
                params.translate      = false;
                params.temperature    = temp;

                if (whisper_full(g_whisperCtx, params,
                                 rollingBuffer.data(), rollingBuffer.size()) == 0) {
                    int n_segments = whisper_full_n_segments(g_whisperCtx);
                    for (int i = lastPrintedSegment; i < n_segments; i++) {
                        const char* text = whisper_full_get_segment_text(g_whisperCtx, i);
                        float conf = 1.0f -
                            whisper_full_get_segment_no_speech_prob(g_whisperCtx, i);
                        std::cout << "[Partial] " << text << " [conf=" << conf << "]\n";
                        g_lastSegmentConfidence = conf;
                        transcript += text;
                        transcript += " ";
                    }
                    lastPrintedSegment = n_segments;
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        auto msSilent =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeech).count();

        if (inSpeech) {
            auto msSpeech =
                std::chrono::duration_cast<std::chrono::milliseconds>(lastSpeech - speechStart).count();

            // 🔹 Softer cutoff: finalize if EITHER condition is met
            if (msSilent > g_silenceTimeoutMs || msSpeech > g_minSpeechMs) {
                std::cout << "[DEBUG][Voice] Finalizing (msSilent=" << msSilent
                          << ", msSpeech=" << msSpeech << ")\n";
                break;
            }
        }

        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (!transcript.empty() && transcript.back() == ' ')
        transcript.pop_back();

    std::cout << "[Voice] Final transcript: " << transcript << "\n";
    return transcript;
}
