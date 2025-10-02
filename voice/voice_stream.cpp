#include "voice_stream.hpp"
#include "console_history.hpp"
#include "ui_helpers.hpp"
#include "commands/commands_core.hpp"
#include "ai/ai.hpp"
#include "nlp/nlp.hpp"
#include "synonyms.hpp"
#include "resources.hpp"
#include "voice.hpp"
#include "logger.hpp"

#include <whisper.h>
#include <portaudio.h>
#include <filesystem>
#include <mutex>
#include <thread>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

// ---------------- Config (from ai_config.json via ai.cpp) ----------------
extern double g_silenceThreshold;
extern int g_silenceTimeoutMs;
extern std::string g_whisperLanguage;
extern int g_whisperMaxTokens;

// ---------------- State ----------------
VoiceStream::State VoiceStream::g_state;

// Minimum buffer before calling Whisper (~100ms at 16kHz)
constexpr size_t MIN_SAMPLES = 1600;

// PCM accumulator for incremental mode
static std::vector<float> pcmAccumulator;

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
    if (buffer.size() <= VoiceStream::g_state.processedSamples) return;

    // Add new samples
    std::vector<float> newAudio(buffer.begin() + VoiceStream::g_state.processedSamples, buffer.end());
    VoiceStream::g_state.processedSamples = buffer.size();
    pcmAccumulator.insert(pcmAccumulator.end(), newAudio.begin(), newAudio.end());

    if (pcmAccumulator.size() < MIN_SAMPLES) {
        std::cout << "[DEBUG][VoiceStream] Accumulating... ("
                  << pcmAccumulator.size() << "/" << MIN_SAMPLES << " samples)\n";
        return;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.no_timestamps = true;
    params.max_tokens = g_whisperMaxTokens;
    params.language = g_whisperLanguage.c_str();

    if (whisper_full(ctx, params, pcmAccumulator.data(), (int)pcmAccumulator.size()) == 0) {
        int n = whisper_full_n_segments(ctx);
        if (n > 0) {
            std::string latest = whisper_full_get_segment_text(ctx, n - 1);
            if (!latest.empty()) {
                VoiceStream::g_state.partial += latest + " ";
                ui_set_textbox(VoiceStream::g_state.partial);
                std::cout << "[VoiceStream] Partial: " << latest << "\n";
            }
        }
    } else {
        std::cerr << "[VoiceStream] ERROR: whisper_full() failed\n";
    }

    pcmAccumulator.clear();
}

// ---------------- Core Loop ----------------
static void run(whisper_context* ctx,
                ConsoleHistory* uiHistory,
                std::vector<Timer>& uiTimers,
                nlohmann::json& uiLongTermMemory,
                NLP& nlp) {
    VoiceStream::g_state.partial.clear();
    VoiceStream::g_state.processedSamples = 0;
    pcmAccumulator.clear();

    if (Pa_Initialize() != paNoError) {
        uiHistory->push("[VoiceStream] ERROR: Failed to initialize PortAudio", sf::Color::Red);
        VoiceStream::g_state.running = false;
        return;
    }

    int deviceIndex = (VoiceStream::g_state.inputDeviceIndex >= 0) ? VoiceStream::g_state.inputDeviceIndex
                                                                   : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice || deviceIndex < 0 || deviceIndex >= Pa_GetDeviceCount()) {
        uiHistory->push("[VoiceStream] ERROR: No valid input device found", sf::Color::Red);
        Pa_Terminate();
        VoiceStream::g_state.running = false;
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
    if (Pa_OpenStream(&stream,
                      &inputParams,
                      nullptr,
                      16000,
                      512,
                      paNoFlag,
                      [](const void* input, void*, unsigned long frameCount,
                         const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* userData) -> int {
                          auto* audio = reinterpret_cast<VoiceStream::State::AudioData*>(userData);
                          const float* in = static_cast<const float*>(input);
                          if (in) {
                              std::lock_guard<std::mutex> lock(audio->mtx);
                              audio->buffer.insert(audio->buffer.end(), in, in + frameCount);
                              audio->ready = true;
                          }
                          return paContinue;
                      },
                      &VoiceStream::g_state.audio) != paNoError || !stream) {
        uiHistory->push("[VoiceStream] ERROR: Could not open mic stream", sf::Color::Red);
        Pa_Terminate();
        VoiceStream::g_state.running = false;
        return;
    }

    if (Pa_StartStream(stream) != paNoError) {
        uiHistory->push("[VoiceStream] ERROR: Could not start mic stream", sf::Color::Red);
        Pa_CloseStream(stream);
        Pa_Terminate();
        VoiceStream::g_state.running = false;
        return;
    }

    uiHistory->push("[VoiceStream] Listening...", sf::Color(0, 200, 255));
    auto lastSpeechTime = std::chrono::steady_clock::now();

    while (VoiceStream::g_state.running) {
        std::vector<float> pcm;
        {
            std::lock_guard<std::mutex> lock(VoiceStream::g_state.audio.mtx);
            if (VoiceStream::g_state.audio.ready) {
                pcm = VoiceStream::g_state.audio.buffer;
                VoiceStream::g_state.audio.ready = false;
                VoiceStream::g_state.audio.buffer.clear();
            }
        }

        if (!pcm.empty()) {
            processPCM(ctx, pcm);

            if (!isSilence(pcm)) {
                lastSpeechTime = std::chrono::steady_clock::now();
            }

            auto now = std::chrono::steady_clock::now();
            auto silenceMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeechTime).count();

            if (!VoiceStream::g_state.partial.empty() && silenceMs > g_silenceTimeoutMs) {
                std::string clean = sanitizeTranscript(VoiceStream::g_state.partial);
                Intent intent = nlp.parse(clean);

                if (intent.matched) {
                    std::cout << "[VoiceStream] Dispatching command: " << intent.name << "\n";
                    handleCommand(clean);
                } else {
                    std::string fullReply;
                    ai_process_stream(
                        VoiceStream::g_state.partial,
                        uiLongTermMemory,
                        [&](const std::string& chunk) {
                            fullReply += chunk;
                            ui_set_textbox(fullReply);
                            std::cout << chunk << std::flush;
                        });
                    uiHistory->push("[AI] " + fullReply, sf::Color::Green);
                }

                VoiceStream::g_state.partial.clear();
                ui_set_textbox("");
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    uiHistory->push("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
}

// ---------------- Control API ----------------
bool VoiceStream::isRunning() {
    return g_state.running;
}

bool VoiceStream::start(whisper_context* ctx,
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

void VoiceStream::stop() {
    if (g_state.running) {
        g_state.running = false;
    }
}

void VoiceStream::calibrateSilence() {
    // TODO: implement real calibration
    std::cout << "[VoiceStream] Calibrating silence threshold (stub)\n";
}

// ---------------- One-shot listenOnce ----------------
std::string Voice::listenOnce() {
    LOG_DEBUG("Voice", "listenOnce() starting…");

    if (Pa_Initialize() != paNoError) {
        LOG_ERROR("Voice", "PortAudio init failed in listenOnce()");
        return "";
    }

    int deviceIndex = Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        LOG_ERROR("Voice", "No valid input device for listenOnce()");
        Pa_Terminate();
        return "";
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    struct TempAudio { std::vector<float> buffer; bool ready=false; std::mutex mtx; } audio;

    auto cb = [](const void* input, void*, unsigned long frameCount,
                 const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void* userData) -> int {
        auto* ad = reinterpret_cast<TempAudio*>(userData);
        const float* in = static_cast<const float*>(input);
        if (in) {
            std::lock_guard<std::mutex> lock(ad->mtx);
            ad->buffer.insert(ad->buffer.end(), in, in + frameCount);
            ad->ready = true;
        }
        return paContinue;
    };

    PaStream* stream = nullptr;
    if (Pa_OpenStream(&stream, &inputParams, nullptr,
                      16000, 512, paNoFlag, cb, &audio) != paNoError || !stream) {
        LOG_ERROR("Voice", "Could not open mic stream in listenOnce()");
        Pa_Terminate();
        return "";
    }

    Pa_StartStream(stream);

    auto lastSpeechTime = std::chrono::steady_clock::now();
    std::vector<float> pcmBuffer;
    std::string transcript;

    while (true) {
        std::vector<float> pcm;
        {
            std::lock_guard<std::mutex> lock(audio.mtx);
            if (audio.ready) {
                pcm = audio.buffer;
                audio.buffer.clear();
                audio.ready = false;
            }
        }

        if (!pcm.empty()) {
            pcmBuffer.insert(pcmBuffer.end(), pcm.begin(), pcm.end());

            if (!isSilence(pcm)) {
                lastSpeechTime = std::chrono::steady_clock::now();
            }

            auto now = std::chrono::steady_clock::now();
            auto silenceMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSpeechTime).count();

            if (silenceMs > g_silenceTimeoutMs && !pcmBuffer.empty()) {
                whisper_context* ctx = Voice::getWhisperContext();
                whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                params.no_timestamps = true;
                params.max_tokens = g_whisperMaxTokens;
                params.language = g_whisperLanguage.c_str();

                if (whisper_full(ctx, params, pcmBuffer.data(), (int)pcmBuffer.size()) == 0) {
                    int n = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n; i++) {
                        transcript += whisper_full_get_segment_text(ctx, i);
                    }
                }
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    transcript = sanitizeTranscript(transcript);
    LOG_DEBUG("Voice", "listenOnce() finished: " + transcript);
    return transcript;
}
