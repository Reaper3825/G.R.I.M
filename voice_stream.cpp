#include "voice_stream.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "commands.hpp"
#include "timer.hpp"
#include "ai.hpp"
#include "nlp.hpp"
#include "synonyms.hpp"

#include <whisper.h>
#include <portaudio.h>
#include <filesystem>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

namespace VoiceStream {

// ---------------- State ----------------
State g_state;

static double g_silenceThreshold = 0.02;
static int g_silenceTimeoutMs = 4000;
static std::string g_whisperLanguage = "en";
static int g_whisperMaxTokens = 32;

// Minimum buffer before calling Whisper (~100ms at 16kHz)
constexpr size_t MIN_SAMPLES = 1600;

// Accumulator for PCM samples
static std::vector<float> pcmAccumulator;

// ---------------- PortAudio Callback ----------------
static int recordCallback(const void* input,
                          void* /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void* userData) {
    auto* audio = reinterpret_cast<State::AudioData*>(userData);
    const float* in = static_cast<const float*>(input);
    if (in) {
        audio->buffer.insert(audio->buffer.end(), in, in + frameCount);
        audio->ready = true;
    }
    return paContinue;
}

// ---------------- Silence Detection ----------------
static bool isSilence(const std::vector<float>& pcm) {
    if (pcm.empty()) return true;
    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();
    double rms = std::sqrt(energy);
    bool silent = rms < g_silenceThreshold;

    std::cout << "[DEBUG][VoiceStream] RMS=" << rms
              << " threshold=" << g_silenceThreshold
              << " -> " << (silent ? "SILENCE" : "VOICE") << "\n";

    return silent;
}

// ---------------- Transcript Sanitizer ----------------
static std::string sanitizeTranscript(const std::string& input) {
    std::string out = input;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    while (!out.empty() && ispunct(out.back())) out.pop_back();
    while (!out.empty() && isspace(out.front())) out.erase(out.begin());
    while (!out.empty() && isspace(out.back())) out.pop_back();
    return out;
}

// ---------------- Whisper Incremental Processing ----------------
static void processPCM(whisper_context* ctx, const std::vector<float>& buffer) {
    if (buffer.size() <= g_state.processedSamples) return;

    // Append new samples to accumulator
    std::vector<float> newAudio(buffer.begin() + g_state.processedSamples, buffer.end());
    g_state.processedSamples = buffer.size();
    pcmAccumulator.insert(pcmAccumulator.end(), newAudio.begin(), newAudio.end());

    if (pcmAccumulator.size() < MIN_SAMPLES) {
        std::cout << "[DEBUG][VoiceStream] Accumulating... ("
                  << pcmAccumulator.size() << "/" << MIN_SAMPLES << " samples)\n";
        return;
    }

    // Pad if still short (safety net)
    if (pcmAccumulator.size() < MIN_SAMPLES) {
        pcmAccumulator.resize(MIN_SAMPLES, 0.0f);
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.no_timestamps = true;
    params.max_tokens = g_whisperMaxTokens;
    params.language = g_whisperLanguage.c_str();

    if (whisper_full(ctx, params, pcmAccumulator.data(), pcmAccumulator.size()) == 0) {
        int n = whisper_full_n_segments(ctx);
        if (n > 0) {
            std::string latest = whisper_full_get_segment_text(ctx, n - 1);
            if (!latest.empty()) {
                g_state.partial += latest + " ";
                ui_set_textbox(g_state.partial);
                std::cout << "[VoiceStream] Partial: " << latest << "\n";
            }
        }
    } else {
        std::cerr << "[VoiceStream] ERROR: whisper_full() failed\n";
    }

    // Reset accumulator after processing
    pcmAccumulator.clear();
}

// ---------------- Core Loop ----------------
static void run(whisper_context* ctx,
                ConsoleHistory* history,
                std::vector<Timer>& timers,
                nlohmann::json& longTermMemory,
                NLP& nlp) {
    g_state.partial.clear();
    g_state.processedSamples = 0;
    pcmAccumulator.clear();

    if (Pa_Initialize() != paNoError) {
        history->push("[VoiceStream] ERROR: Failed to initialize PortAudio", sf::Color::Red);
        g_state.running = false;
        return;
    }

    int deviceIndex = (g_state.inputDeviceIndex >= 0) ? g_state.inputDeviceIndex : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        history->push("[VoiceStream] ERROR: No input device found", sf::Color::Red);
        Pa_Terminate();
        g_state.running = false;
        return;
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaStream* stream = nullptr;
    if (Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 512, paNoFlag, recordCallback, &g_state.audio) != paNoError || !stream) {
        history->push("[VoiceStream] ERROR: Could not open mic stream", sf::Color::Red);
        Pa_Terminate();
        g_state.running = false;
        return;
    }

    if (Pa_StartStream(stream) != paNoError) {
        history->push("[VoiceStream] ERROR: Could not start mic stream", sf::Color::Red);
        Pa_CloseStream(stream);
        Pa_Terminate();
        g_state.running = false;
        return;
    }

    history->push("[VoiceStream] Listening...", sf::Color(0, 200, 255));
    auto lastSpeechTime = std::chrono::steady_clock::now();

    while (g_state.running) {
        std::vector<float> pcm;
        if (g_state.audio.ready) {
            pcm = g_state.audio.buffer;
            g_state.audio.ready = false;
            g_state.audio.buffer.clear();
        }

        if (!pcm.empty()) {
            processPCM(ctx, pcm);

            if (!isSilence(pcm)) {
                lastSpeechTime = std::chrono::steady_clock::now();
            }

            auto now = std::chrono::steady_clock::now();
            auto silenceMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeechTime).count();

            if (!g_state.partial.empty() && silenceMs > g_silenceTimeoutMs) {
                std::string clean = sanitizeTranscript(g_state.partial);
                Intent intent = nlp.parse(clean);

                if (intent.matched) {
                    auto currentDir = fs::current_path();
                    handleCommand(intent, g_state.partial, currentDir, timers, longTermMemory, nlp, *history);
                } else {
                    std::string fullReply;
                    ai_process_stream(g_state.partial, longTermMemory,
                        [&](const std::string& chunk) {
                            fullReply += chunk;
                            ui_set_textbox(fullReply);
                            std::cout << chunk << std::flush;
                        }
                    );
                    history->push("[AI] " + fullReply, sf::Color::Green);
                }

                g_state.partial.clear();
                ui_set_textbox("");
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    history->push("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
}

// ---------------- Control API ----------------
bool isRunning() {
    return g_state.running;
}

bool start(whisper_context* ctx,
           ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp) {
    if (g_state.running) {
        history->push("[VoiceStream] Already running", sf::Color::Yellow);
        return false;
    }

    g_state.running = true;

    std::thread([=, &timers, &longTermMemory, &nlp]() mutable {
        run(ctx, history, timers, longTermMemory, nlp);
    }).detach();

    return true;
}

void stop() {
    g_state.running = false;
}

// ---------------- Calibration ----------------
void calibrateSilence() {
    std::cout << "[Calibration] (stub) Silence calibration not yet implemented.\n";
}

} // namespace VoiceStream
