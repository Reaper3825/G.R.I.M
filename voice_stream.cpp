#include "voice_stream.hpp"
#include "nlp.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include <whisper.h>
#include <portaudio.h>
#include "commands.hpp"
#include "timer.hpp"
#include "ai.hpp"
#include "synonyms.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <csignal>
#include <chrono>
#include <cmath>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

#define SAMPLE_RATE       16000
#define FRAMES_PER_BUFFER 512
#define CHUNK_SIZE        (SAMPLE_RATE / 2) // 0.5 seconds

// ---------------- Audio Data ----------------
struct AudioData {
    std::vector<float> buffer;
    std::mutex mutex;
    bool ready = false;
};

static AudioData g_audio;
static std::atomic<bool> g_running{false};
static std::string g_partial; // evolving transcript
static size_t g_processedSamples = 0;

// ---------------- Globals ----------------
extern double g_silenceThreshold;        // from ai_config.json or calibration
extern float g_lastSegmentConfidence;    // from voice.cpp
extern int g_silenceTimeoutMs;           // from ai_config.json
extern std::string g_whisperLanguage;    // from ai_config.json
extern int g_whisperMaxTokens;           // from ai_config.json
extern nlohmann::json aiConfig;          // full config
extern nlohmann::json longTermMemory;    // persistent memory

static int g_inputDeviceIndex = -1;      // mic selection

// ---------------- Signal Handling ----------------
static void handleSigint(int) {
    g_running = false;
}

// ---------------- PortAudio Callback ----------------
static int recordCallback(const void *input,
                          void * /*output*/,
                          unsigned long frameCount,
                          const PaStreamCallbackTimeInfo* /*timeInfo*/,
                          PaStreamCallbackFlags /*statusFlags*/,
                          void * /*userData*/) {
    const float *in = static_cast<const float*>(input);
    if (in) {
        std::lock_guard<std::mutex> lock(g_audio.mutex);
        g_audio.buffer.insert(g_audio.buffer.end(), in, in + frameCount);
        g_audio.ready = true;
    }
    return paContinue;
}

// ---------------- Silence Detection (RMS) ----------------
static bool isSilence(const std::vector<float> &pcm) {
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

// ---------------- Completion Heuristics ----------------
#include "synonyms.hpp"
static bool looksComplete(const std::string& text) {
    if (text.empty()) return false;
    std::string trimmed = text;
    while (!trimmed.empty() && isspace(trimmed.back())) trimmed.pop_back();
    if (trimmed.empty()) return false;
    char last = trimmed.back();
    if (last == '.' || last == '?' || last == '!') return true;
    std::istringstream iss(trimmed);
    std::string word, lastWord;
    while (iss >> word) lastWord = word;
    for (auto& w : g_completionTriggers) {
        if (lastWord == w) return true;
        auto it = g_synonyms.find(w);
        if (it != g_synonyms.end()) {
            for (auto& syn : it->second) {
                if (lastWord == syn) return true;
            }
        }
    }
    return false;
}

// ---------------- Whisper Incremental Processing ----------------
static void processPCM(whisper_context* ctx, const std::vector<float>& buffer) {
    if (buffer.size() <= g_processedSamples) return;
    std::vector<float> newAudio(buffer.begin() + g_processedSamples, buffer.end());
    g_processedSamples = buffer.size();

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime   = false;
    params.print_progress   = false;
    params.translate        = false;
    params.no_context       = false;
    params.single_segment   = false;
    params.no_timestamps    = true;
    params.token_timestamps = false;
    params.max_tokens       = g_whisperMaxTokens;
    params.language         = g_whisperLanguage.c_str();

    if (whisper_full(ctx, params, newAudio.data(), newAudio.size()) == 0) {
        int n = whisper_full_n_segments(ctx);
        if (n > 0) {
            std::string latest = whisper_full_get_segment_text(ctx, n - 1);
            if (!latest.empty()) {
                g_partial += latest + " ";
                ui_set_textbox(g_partial);
                std::cout << "[VoiceStream] Partial: " << latest << "\n";
            }
        }
    }
}

// ---------------- Main Voice Stream ----------------
void runVoiceStream(whisper_context *ctx,
                    ConsoleHistory* history,
                    std::vector<Timer>& timers,
                    nlohmann::json& longTermMemory,
                    NLP& nlp) {
    g_running = true;
    g_partial.clear();
    g_processedSamples = 0;
    signal(SIGINT, handleSigint);

    // Load device index
    g_inputDeviceIndex = aiConfig.value("input_device_index", -1);

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        history->push("[VoiceStream] ERROR: Failed to initialize PortAudio", sf::Color::Red);
        return;
    }

    // Select mic
    int deviceIndex = (g_inputDeviceIndex >= 0) ? g_inputDeviceIndex : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        history->push("[VoiceStream] ERROR: No input device found", sf::Color::Red);
        Pa_Terminate();
        return;
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    std::cout << "[VoiceStream] Using input device #" << deviceIndex
              << " (" << devInfo->name << ")\n";

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaStream *stream = nullptr;
    err = Pa_OpenStream(&stream,
                        &inputParams, nullptr,
                        SAMPLE_RATE,
                        FRAMES_PER_BUFFER,
                        paNoFlag,
                        recordCallback,
                        nullptr);
    if (err != paNoError || !stream) {
        history->push("[VoiceStream] ERROR: Could not open mic stream", sf::Color::Red);
        Pa_Terminate();
        return;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        history->push("[VoiceStream] ERROR: Could not start mic stream", sf::Color::Red);
        Pa_CloseStream(stream);
        Pa_Terminate();
        return;
    }

    history->push("[VoiceStream] Listening... (Ctrl+C to stop)", sf::Color(0, 200, 255));
    std::cout << "[VoiceStream] Listening... speak into your mic (Ctrl+C to stop)\n";

    auto lastSpeechTime = std::chrono::steady_clock::now();

    std::thread worker([&]() {
        while (g_running) {
            std::vector<float> pcm;
            {
                std::lock_guard<std::mutex> lock(g_audio.mutex);
                if (!g_audio.ready) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                pcm = g_audio.buffer;
                g_audio.ready = false;
            }

            processPCM(ctx, pcm);

            if (!isSilence(pcm)) {
                lastSpeechTime = std::chrono::steady_clock::now();
            }

            auto now = std::chrono::steady_clock::now();
            auto silenceMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeechTime).count();

            if (!g_partial.empty() && silenceMs > g_silenceTimeoutMs) {
                std::cout << "[DEBUG][VoiceStream] Silence timeout (" << silenceMs << " ms), finalizing.\n";

                history->push("[VoiceStream Heard] " + g_partial, sf::Color::Yellow);

                std::string clean = sanitizeTranscript(g_partial);
                Intent intent = nlp.parse(clean);

                std::cout << "[DEBUG][VoiceStream] Intent parsed: name='" << intent.name
                          << "', matched=" << intent.matched
                          << ", slots=" << intent.slots.size()
                          << ", score=" << intent.score << "\n";

                if (intent.matched) {
                    auto currentDir = fs::current_path();
                    handleCommand(intent, g_partial, currentDir,
                                  timers, longTermMemory, nlp, *history);
                } else {
                    std::string fullReply;
                    ai_process_stream(g_partial, longTermMemory,
                        [&](const std::string& chunk) {
                            fullReply += chunk;
                            ui_set_textbox(fullReply);
                            std::cout << chunk << std::flush;
                        }
                    );
                    history->push("[AI] " + fullReply, sf::Color::Green);
                    std::cout << "\n[VoiceStream] AI response complete.\n";
                }

                g_partial.clear();
                ui_set_textbox("");
                std::cout << "[VoiceStream] Command committed.\n";
            }
        }
    });

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    worker.join();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    history->push("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
    std::cout << "[VoiceStream] Stopped.\n";
}

// ---------------- Calibration ----------------
void calibrateSilence() {
    std::cout << "[Calibration] Starting 2s silence calibration...\n";

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "[Calibration] PortAudio init failed\n";
        return;
    }

    int deviceIndex = (g_inputDeviceIndex >= 0) ? g_inputDeviceIndex : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        std::cerr << "[Calibration] No input device found\n";
        Pa_Terminate();
        return;
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    std::cout << "[Calibration] Using input device #" << deviceIndex
              << " (" << devInfo->name << ")\n";

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    PaStream *stream = nullptr;
    std::vector<float> buffer;

    err = Pa_OpenStream(&stream,
                        &inputParams, nullptr,
                        SAMPLE_RATE,
                        FRAMES_PER_BUFFER,
                        paNoFlag,
                        recordCallback,
                        nullptr);
    if (err != paNoError || !stream) {
        std::cerr << "[Calibration] Could not open mic stream\n";
        Pa_Terminate();
        return;
    }

    Pa_StartStream(stream);
    auto start = std::chrono::steady_clock::now();

    while (true) {
        {
            std::lock_guard<std::mutex> lock(g_audio.mutex);
            if (!g_audio.buffer.empty()) {
                buffer.insert(buffer.end(), g_audio.buffer.begin(), g_audio.buffer.end());
                g_audio.buffer.clear();
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= 2) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (buffer.empty()) {
        std::cerr << "[Calibration] No mic input captured\n";
        return;
    }

    double energy = 0.0;
    for (float s : buffer) energy += s * s;
    energy /= buffer.size();
    double rms = std::sqrt(energy);

    double multiplier = 3.0;
    if (aiConfig.contains("silence_detection") &&
        aiConfig["silence_detection"].is_object()) {
        multiplier = aiConfig["silence_detection"].value("auto_multiplier", 3.0);
    }

    g_silenceThreshold = rms * multiplier;
    longTermMemory["voice_baseline"] = rms;

    std::cout << "[Calibration] New silence threshold (auto): " << g_silenceThreshold
              << " (baseline=" << rms << ", multiplier=" << multiplier << ")\n";
    saveMemory();
}
