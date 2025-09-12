#include "voice.hpp"
#include "resources.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <portaudio.h>
#include <chrono>
#include <cmath>
#include "whisper.h"
#include <nlohmann/json.hpp>

#define SAMPLE_RATE        16000
#define FRAMES_PER_BUFFER   512
#define CHUNK_SIZE         (SAMPLE_RATE / 2) // 0.5 seconds

// 🔹 Global Whisper context (definition for extern in voice.hpp)
struct whisper_context *g_whisperCtx = nullptr;

// 🔹 Silence threshold from ai_config.json
extern double g_silenceThreshold;

// 🔹 Environment baseline (auto-calibrated and cached in memory)
static double g_envLevel = 0.0;

// ---------------- Audio Recording ----------------
struct AudioData {
    std::vector<float> buffer;
    bool ready = false;
};

static int recordCallback(const void *input,
                          void * /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void *userData) {
    AudioData *data = reinterpret_cast<AudioData*>(userData);
    const float *in = reinterpret_cast<const float*>(input);

    if (in) {
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
    }

    return paContinue; // keep streaming
}

// ---------------- Silence Detection ----------------
static bool isSilence(const std::vector<float> &pcm) {
    if (pcm.empty()) return true;

    // Calculate RMS of the chunk
    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    double rms = std::sqrt(energy);

    // subtract baseline (like zeroing a scale)
    double adjusted = std::max(0.0, rms - g_envLevel);

    std::cout << "[DEBUG][Voice] RMS=" << rms
              << " baseline=" << g_envLevel
              << " adjusted=" << adjusted
              << " threshold=" << g_silenceThreshold << "\n";

    return adjusted < g_silenceThreshold;
}

// ---------------- Whisper Init ----------------
bool initWhisper() {
    namespace fs = std::filesystem;
    fs::path baseModel = fs::path(getResourcePath()) / "models" / "whisper" / "ggml-base.en.bin";
    fs::path smallModel = fs::path(getResourcePath()) / "models" / "whisper" / "ggml-small.bin";

    fs::path modelPathFS;
    if (fs::exists(baseModel)) {
        modelPathFS = baseModel;
        std::cout << "[Voice] Using base.en model: " << modelPathFS << std::endl;
    } else if (fs::exists(smallModel)) {
        modelPathFS = smallModel;
        std::cout << "[Voice] Base.en not found, using small model: " << modelPathFS << std::endl;
    } else {
        std::cerr << "[Voice] No Whisper model found in models/ !" << std::endl;
        return false;
    }

    struct whisper_context_params wparams = whisper_context_default_params();
    g_whisperCtx = whisper_init_from_file_with_params(modelPathFS.string().c_str(), wparams);

    if (!g_whisperCtx) {
        std::cerr << "[Voice] Failed to load Whisper model: " << modelPathFS << std::endl;
        return false;
    }

    std::cout << "[Voice] Whisper initialized successfully." << std::endl;
    return true;
}

// ---------------- Voice Demo ----------------
std::string runVoiceDemo(const std::string & /*modelPath*/,
                         nlohmann::json& longTermMemory) {  // ⬅ pass memory reference
    if (!g_whisperCtx) {
        std::cerr << "[Voice] Whisper not initialized! Call initWhisper() first." << std::endl;
        return "";
    }

    // 🔹 Load cached baseline if available
    if (longTermMemory.contains("voice_baseline")) {
        g_envLevel = longTermMemory["voice_baseline"].get<double>();
        std::cout << "[Voice] Loaded cached environment baseline: " << g_envLevel << "\n";
    } else {
        std::cout << "[Voice] No cached baseline, will auto-calibrate...\n";
    }

    // Initialize PortAudio
    if (Pa_Initialize() != paNoError) {
        std::cerr << "[Voice] Failed to initialize PortAudio" << std::endl;
        return "[ERROR] Failed to init PortAudio";
    }
    std::cout << "[DEBUG][Voice] PortAudio initialized\n";

    AudioData data;
    PaStream *stream;

    if (Pa_OpenDefaultStream(&stream,
                             1, // mono input
                             0, // no output
                             paFloat32,
                             SAMPLE_RATE,
                             FRAMES_PER_BUFFER,
                             recordCallback,
                             &data) != paNoError) {
        std::cerr << "[Voice] Failed to open audio stream" << std::endl;
        Pa_Terminate();
        return "[ERROR] Failed to open audio stream";
    }

    std::cout << "[Voice] Listening... speak into the mic." << std::endl;
    Pa_StartStream(stream);
    std::cout << "[DEBUG][Voice] Stream started (sampleRate=" << SAMPLE_RATE
              << ", buffer=" << FRAMES_PER_BUFFER << ")\n";

    std::vector<float> rollingBuffer;
    std::string transcript;
    int lastPrintedSegment = 0;

    auto lastSpeech = std::chrono::steady_clock::now();

    while (true) {
        // Process 0.5s chunks
        if (data.buffer.size() >= CHUNK_SIZE) {
            std::vector<float> chunk(data.buffer.begin(),
                                     data.buffer.begin() + CHUNK_SIZE);
            data.buffer.erase(data.buffer.begin(),
                              data.buffer.begin() + CHUNK_SIZE);

            // Calculate RMS for this chunk
            double energy = 0.0;
            for (float sample : chunk) energy += sample * sample;
            energy /= chunk.size();
            double rms = std::sqrt(energy);
            double dB = 20.0 * log10(rms + 1e-6);
            std::cout << "[DEBUG][Voice] Chunk loudness: RMS=" << rms
                      << " dB=" << dB << "\n";

            // 🔹 Auto-calibrate baseline (first ~1s if not cached)
            if (g_envLevel == 0.0 && rollingBuffer.size() < SAMPLE_RATE) {
                g_envLevel = rms;
                longTermMemory["voice_baseline"] = g_envLevel;
                std::cout << "[Voice] Environment baseline calibrated: " << g_envLevel << "\n";
            }

            if (!isSilence(chunk)) {
                lastSpeech = std::chrono::steady_clock::now();
                rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
            }

            // Limit rolling buffer to ~10s of audio
            if (rollingBuffer.size() > SAMPLE_RATE * 10) {
                rollingBuffer.erase(rollingBuffer.begin(),
                                    rollingBuffer.begin() + (rollingBuffer.size() - SAMPLE_RATE * 10));
                lastPrintedSegment = 0; // reset for fresh decode
            }

            // Run Whisper if we have at least 2s of audio
            if (rollingBuffer.size() >= SAMPLE_RATE * 2) {
                // 🔹 Normalize audio before Whisper
                float maxVal = 0.0f;
                for (float s : rollingBuffer) maxVal = std::max(maxVal, std::fabs(s));
                if (maxVal > 0.0f) {
                    for (float &s : rollingBuffer) s /= maxVal;
                }

                whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                params.language = "en";          // force English
                params.no_context = false;       // keep context
                params.single_segment = false;   // allow multiple segments
                params.print_realtime = false;
                params.print_progress = false;
                params.translate = false;

                if (whisper_full(g_whisperCtx, params,
                                 rollingBuffer.data(),
                                 rollingBuffer.size()) == 0) {
                    int n_segments = whisper_full_n_segments(g_whisperCtx);

                    for (int i = lastPrintedSegment; i < n_segments; i++) {
    const char *text = whisper_full_get_segment_text(g_whisperCtx, i);
    float conf = whisper_full_get_segment_avg_logprob(g_whisperCtx, i);

    std::cout << "[Partial] " << text 
              << " [conf=" << conf << "]" << std::endl;

    transcript += text;
    transcript += " ";

    // Optional: mark this segment as "complete" if confidence is strong
    if (conf > -0.3) {
        // You could trigger early cutoff logic here
        // e.g., finalize faster in voice_stream.cpp
    }
}

                    lastPrintedSegment = n_segments;
                } else {
                    std::cerr << "[Voice] Whisper inference failed" << std::endl;
                }
            }
        }

        // Stop if silence for >4s after speech
        auto now = std::chrono::steady_clock::now();
        auto msSilent = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeech).count();
        if (msSilent > 4000 && !transcript.empty()) {
            std::cout << "[DEBUG][Voice] Silence timeout reached (" << msSilent
                      << " ms), finalizing transcript.\n";
            break;
        }

        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cout << "[DEBUG][Voice] PortAudio shutdown complete\n";

    if (!transcript.empty() && transcript.back() == ' ')
        transcript.pop_back();

    std::cout << "[Voice] Final transcript: " << transcript << std::endl;
    return transcript;
}
