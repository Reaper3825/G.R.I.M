#include "voice.hpp"
#include "resources.hpp"
#include "whisper.h"
#include "ai.hpp"   // ✅ brings in extern globals (silenceThreshold, silenceTimeoutMs)

#include <iostream>
#include <vector>
#include <filesystem>
#include <portaudio.h>
#include <chrono>
#include <cmath>
#include <fstream>
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

// Config values (set from ai_config.json)
extern nlohmann::json aiConfig;
extern nlohmann::json longTermMemory;

int g_minSpeechMs  = 500;
int g_minSilenceMs = 1200;
int g_inputDeviceIndex = -1; // -1 = use default

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

    return paContinue;
}

// ------------------------------------------------------------
// Silence Detection (RMS)
// ------------------------------------------------------------
static bool isSilence(const std::vector<float>& pcm) {
    if (pcm.empty()) return true;

    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    double rms = std::sqrt(energy);

    bool silent = rms < g_silenceThreshold;

    std::cout << "[DEBUG][Voice] RMS=" << rms
              << " baseline=" << g_envLevel
              << " threshold=" << g_silenceThreshold
              << " -> " << (silent ? "SILENCE" : "VOICE") << "\n";

    return silent;
}

// ------------------------------------------------------------
// Save WAV (debugging)
// ------------------------------------------------------------
void saveWav(const std::string& path, const std::vector<float>& pcm, int sampleRate = SAMPLE_RATE) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;

    int dataSize = pcm.size() * sizeof(int16_t);
    int fileSize = 36 + dataSize;

    // RIFF header
    f.write("RIFF", 4);
    f.write((char*)&fileSize, 4);
    f.write("WAVE", 4);

    // fmt chunk
    int fmtSize = 16;
    short audioFormat = 1;
    short numChannels = 1;
    int byteRate = sampleRate * numChannels * 2;
    short blockAlign = numChannels * 2;
    short bitsPerSample = 16;

    f.write("fmt ", 4);
    f.write((char*)&fmtSize, 4);
    f.write((char*)&audioFormat, 2);
    f.write((char*)&numChannels, 2);
    f.write((char*)&sampleRate, 4);
    f.write((char*)&byteRate, 4);
    f.write((char*)&blockAlign, 2);
    f.write((char*)&bitsPerSample, 2);

    // data chunk
    f.write("data", 4);
    f.write((char*)&dataSize, 4);

    for (float s : pcm) {
        int16_t v = (int16_t)(std::max(-1.0f, std::min(1.0f, s)) * 32767);
        f.write((char*)&v, 2);
    }
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

    // --- Load config ---
    g_inputDeviceIndex = aiConfig.value("input_device_index", -1);
    g_silenceTimeoutMs = aiConfig.value("silence_timeout_ms", 4000);

    if (aiConfig.contains("whisper") && aiConfig["whisper"].is_object()) {
        auto& w = aiConfig["whisper"];
        g_minSpeechMs  = w.value("min_speech_ms", 500);
        g_minSilenceMs = w.value("min_silence_ms", 1200);
    }

    if (aiConfig.contains("silence_detection") &&
        aiConfig["silence_detection"].is_object()) {
        auto& sd = aiConfig["silence_detection"];
        std::string mode = sd.value("mode", "fixed");
        if (mode == "auto") {
            if (longTermMemory.contains("voice_baseline")) {
                g_envLevel = longTermMemory["voice_baseline"].get<double>();
            }
            double mult = sd.value("auto_multiplier", 3.0);
            g_silenceThreshold = std::max(0.00001, g_envLevel * mult);
            std::cout << "[Voice] Auto silence threshold set: "
                      << g_silenceThreshold << " (baseline=" << g_envLevel << ")\n";
        } else {
            g_silenceThreshold = sd.value("silence_threshold", 0.02);
            std::cout << "[Voice] Fixed silence threshold: " << g_silenceThreshold << "\n";
        }
    }

    std::cout << "[Voice] Whisper initialized. "
              << "silence_timeout=" << g_silenceTimeoutMs
              << ", min_speech_ms=" << g_minSpeechMs
              << ", min_silence_ms=" << g_minSilenceMs
              << ", input_device_index=" << g_inputDeviceIndex << "\n";

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

    if (Pa_Initialize() != paNoError) {
        return "[ERROR] Failed to init PortAudio";
    }

    AudioData data;
    PaStream* stream;

    int deviceIndex = (g_inputDeviceIndex >= 0) ? g_inputDeviceIndex : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        Pa_Terminate();
        return "[ERROR] No input device found";
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    std::cout << "[Voice] Using input device #" << deviceIndex
              << " (" << devInfo->name << ")\n";

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    if (Pa_OpenStream(&stream,
                      &inputParams, nullptr,
                      SAMPLE_RATE,
                      FRAMES_PER_BUFFER,
                      paNoFlag,
                      recordCallback,
                      &data) != paNoError) {
        Pa_Terminate();
        return "[ERROR] Failed to open audio stream";
    }

    Pa_StartStream(stream);
    std::cout << "[Voice] Listening...\n";

    std::vector<float> rollingBuffer;
    std::string transcript;

    auto lastSpeech = std::chrono::steady_clock::now();
    auto speechStart = std::chrono::steady_clock::now();
    bool inSpeech = false;
    int silentChunks = 0;

    while (true) {
        if (data.buffer.size() >= CHUNK_SIZE) {
            std::vector<float> chunk(data.buffer.begin(), data.buffer.begin() + CHUNK_SIZE);
            data.buffer.erase(data.buffer.begin(), data.buffer.begin() + CHUNK_SIZE);

            if (!isSilence(chunk)) {
                if (!inSpeech) {
                    speechStart = std::chrono::steady_clock::now();
                    inSpeech = true;
                    std::cout << "[DEBUG][Voice] Speech started\n";
                }
                lastSpeech = std::chrono::steady_clock::now();
                rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
                silentChunks = 0;
            } else if (inSpeech) {
                silentChunks++;
                auto msSinceSpeech =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - lastSpeech).count();
                auto msSpeech =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        lastSpeech - speechStart).count();

                std::cout << "[DEBUG][Voice] Checking finalize -> "
                          << "msSinceSpeech=" << msSinceSpeech
                          << ", msSpeech=" << msSpeech
                          << ", silentChunks=" << silentChunks << "\n";

                if (msSinceSpeech >= g_minSilenceMs &&
                    msSpeech >= g_minSpeechMs &&
                    rollingBuffer.size() > SAMPLE_RATE * 1) {
                    std::cout << "[DEBUG][Voice] Finalize condition met!\n";
                    break;
                }

                if (msSinceSpeech >= g_silenceTimeoutMs) {
                    std::cout << "[DEBUG][Voice] Force finalizing after silence timeout (" 
                              << msSinceSpeech << " ms)\n";
                    break;
                }
            }
        }

        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    // --- Run Whisper ---
    if (!rollingBuffer.empty()) {
        std::cout << "[DEBUG] Sending " << rollingBuffer.size() << " samples to Whisper\n";

        saveWav("captured.wav", rollingBuffer);

        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.print_progress   = false;
        wparams.print_realtime   = false;
        wparams.print_timestamps = true;

        int ret = whisper_full(g_whisperCtx, wparams, rollingBuffer.data(), rollingBuffer.size());
        if (ret != 0) {
            std::cerr << "[Voice] ERROR: Whisper processing failed (code " << ret << ")\n";
        } else {
            int n = whisper_full_n_segments(g_whisperCtx);
            std::cout << "[DEBUG] Whisper returned " << n << " segments\n";
            for (int i = 0; i < n; i++) {
                const char* txt = whisper_full_get_segment_text(g_whisperCtx, i);
                std::cout << "[DEBUG] Segment " << i << ": " << txt << "\n";
                transcript += txt;
                transcript += " ";
            }
        }
    }

    if (!transcript.empty() && transcript.back() == ' ')
        transcript.pop_back();

    std::cout << "[Voice] Final transcript: " << transcript << "\n";
    return transcript;
}
