#include "voice.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <portaudio.h>
#include <chrono>
#include <cmath>
#include "whisper.h"

#define SAMPLE_RATE        16000
#define FRAMES_PER_BUFFER   512

// 🔹 Global Whisper context (definition for extern in voice.hpp)
struct whisper_context *g_whisperCtx = nullptr;

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

    if (input != nullptr) {
        data->buffer.insert(data->buffer.end(), in, in + frameCount);
        std::cout << "[DEBUG][Voice] Captured " << frameCount
                  << " frames (" << data->buffer.size() << " total in buffer)\n";
    }

    return paContinue; // keep streaming
}

static bool isSilence(const std::vector<float> &pcm) {
    if (pcm.empty()) return true;
    double sum = 0.0;
    for (float sample : pcm) {
        sum += std::fabs(sample);
    }
    double avg = sum / pcm.size();
    std::cout << "[DEBUG][Voice] Silence check: avg=" << avg << "\n";
    return avg < 0.01; // tweak threshold if needed
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
std::string runVoiceDemo(const std::string & /*modelPath*/) {
    if (!g_whisperCtx) {
        std::cerr << "[Voice] Whisper not initialized! Call initWhisper() first." << std::endl;
        return "";
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
        // Process ~0.25s chunks
        if (data.buffer.size() >= SAMPLE_RATE / 4) {
            std::vector<float> chunk(data.buffer.begin(),
                                     data.buffer.begin() + SAMPLE_RATE / 4);
            data.buffer.erase(data.buffer.begin(),
                              data.buffer.begin() + SAMPLE_RATE / 4);

            std::cout << "[DEBUG][Voice] Got chunk of " << chunk.size()
                      << " samples from buffer\n";

            if (!isSilence(chunk)) {
                lastSpeech = std::chrono::steady_clock::now();
                rollingBuffer.insert(rollingBuffer.end(), chunk.begin(), chunk.end());
                std::cout << "[DEBUG][Voice] Added to rolling buffer, size="
                          << rollingBuffer.size() << "\n";
            }

            // Limit rolling buffer to ~5s of audio
            if (rollingBuffer.size() > SAMPLE_RATE * 5) {
                rollingBuffer.erase(rollingBuffer.begin(),
                                    rollingBuffer.begin() + (rollingBuffer.size() - SAMPLE_RATE * 5));
                std::cout << "[DEBUG][Voice] Rolling buffer trimmed to "
                          << rollingBuffer.size() << " samples\n";
            }

            if (!rollingBuffer.empty()) {
                whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                params.language = "en";          // force English
                params.no_context = false;       // keep context
                params.single_segment = false;   // multiple partials
                params.print_realtime = false;
                params.print_progress = false;
                params.translate = false;

                std::cout << "[DEBUG][Voice] Running Whisper on "
                          << rollingBuffer.size() << " samples\n";

                if (whisper_full(g_whisperCtx, params, rollingBuffer.data(), rollingBuffer.size()) == 0) {
                    int n_segments = whisper_full_n_segments(g_whisperCtx);
                    std::cout << "[DEBUG][Voice] Whisper returned "
                              << n_segments << " segments\n";

                    for (int i = lastPrintedSegment; i < n_segments; i++) {
                        const char *text = whisper_full_get_segment_text(g_whisperCtx, i);
                        std::cout << "[Partial] " << text << std::endl;
                        transcript += text;
                        transcript += " ";
                    }
                    lastPrintedSegment = n_segments;
                } else {
                    std::cerr << "[Voice] Whisper inference failed" << std::endl;
                }
            }
        }

        // stop if silence for >1.2s after having spoken
        auto now = std::chrono::steady_clock::now();
        auto msSilent = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeech).count();
        if (msSilent > 1200 && !transcript.empty()) {
            std::cout << "[DEBUG][Voice] Silence timeout reached (" << msSilent
                      << " ms), finalizing transcript.\n";
            break;
        }

        Pa_Sleep(50); // tighter loop for snappier updates
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cout << "[DEBUG][Voice] PortAudio shutdown complete\n";

    if (!transcript.empty() && transcript.back() == ' ') {
        transcript.pop_back();
    }

    std::cout << "[Voice] Final transcript: " << transcript << std::endl;
    return transcript;
}
