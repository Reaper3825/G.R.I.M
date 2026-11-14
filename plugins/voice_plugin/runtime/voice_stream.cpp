#include "voice_stream.hpp"
#include "runtime_state.hpp"
#include "voice.hpp"
#include "plugins/plugin_logging.hpp"
#include <whisper.h>
#include <portaudio.h>
#include <filesystem>
#include <mutex>
#include <thread>
#include <cctype>
#include <cmath>

namespace fs = std::filesystem;

// ---------------- Config (from ai_config.json via ai.cpp) ----------------
static double g_silenceThreshold = 0.02;
static int g_silenceTimeoutMs = 1200;
static std::string g_whisperLanguage = "en";
static int g_whisperMaxTokens = 128;

// ---------------- State ----------------
VoiceStream::State VoiceStream::g_state;

// Minimum buffer before calling Whisper (~100ms at 16kHz)
constexpr size_t MIN_SAMPLES = 1600;

// PCM accumulator for incremental mode
static std::vector<float> pcmAccumulator;
//----------------------------------------------------
// ---------------- Silence Detection ----------------
static bool isSilence(const std::vector<float>& pcm) {
    if (pcm.empty()) return true;

    double energy = 0.0;
    for (float s : pcm) energy += s * s;
    energy /= pcm.size();

    double rms = std::sqrt(energy);
    bool silent = rms < g_silenceThreshold;

    LOG_DEBUG("VoiceStream",
        "RMS=" + std::to_string(rms) +
        " threshold=" + std::to_string(g_silenceThreshold) +
        " → " + (silent ? "SILENCE" : "VOICE"));

    return silent;
}

// ---------------- Transcript Sanitizer ----------------
static std::string sanitizeTranscript(const std::string& input) {
    std::string out = input;

    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) {
        out.erase(out.begin());
    }
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
        out.pop_back();
    }

    return out;
}

// ---------------- Whisper Incremental Processing ----------------
static void processPCM(whisper_context* ctx, const std::vector<float>& buffer) {
    pcmAccumulator.insert(pcmAccumulator.end(), buffer.begin(), buffer.end());

    if (pcmAccumulator.size() < MIN_SAMPLES) {
        LOG_DEBUG("VoiceStream", "Accumulating (" + std::to_string(pcmAccumulator.size()) +
                  "/" + std::to_string(MIN_SAMPLES) + " samples)");
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
                // Note: Console now uses ConsolePanel with UIInputBox - no direct buffer access
                LOG_DEBUG("VoiceStream", "Partial recognized: " + latest);
            }
        }
    } else {
        LOG_ERROR("VoiceStream", "whisper_full() failed during incremental processing.");
    }

    pcmAccumulator.clear();
}

// ---------------- Core Loop ----------------
static void run(whisper_context* ctx)
{
    LOG_DEBUG("VoiceStream", "Run() thread entered.");
    LOG_DEBUG("Voice", "VoiceStream thread active");

    VoiceStream::g_state.partial.clear();
    VoiceStream::g_state.processedSamples = 0;
    pcmAccumulator.clear();

    LOG_DEBUG("VoiceStream", "Initializing PortAudio...");
    if (Pa_Initialize() != paNoError) {
        LOG_ERROR("VoiceStream", "Failed to initialize PortAudio!");
        VoiceStream::g_state.running = false;
        return;
    }

    int deviceIndex = (VoiceStream::g_state.inputDeviceIndex >= 0)
                        ? VoiceStream::g_state.inputDeviceIndex
                        : Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice || deviceIndex < 0 || deviceIndex >= Pa_GetDeviceCount()) {
        LOG_ERROR("VoiceStream", "No valid input device found!");
        Pa_Terminate();
        VoiceStream::g_state.running = false;
        return;
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    LOG_DEBUG("VoiceStream", std::string("Using input device: ") + devInfo->name);

    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    LOG_DEBUG("VoiceStream", "Opening input stream...");
    PaStream* stream = nullptr;
    PaError openErr = Pa_OpenStream(&stream,
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
                                    &VoiceStream::g_state.audio);

    if (openErr != paNoError || !stream) {
        LOG_ERROR("VoiceStream", std::string("Could not open mic stream: ") + Pa_GetErrorText(openErr));
        Pa_Terminate();
        VoiceStream::g_state.running = false;
        return;
    }

    LOG_DEBUG("VoiceStream", "Starting input stream...");
    PaError startErr = Pa_StartStream(stream);
    if (startErr != paNoError) {
        LOG_ERROR("VoiceStream", std::string("Failed to start PortAudio stream: ") + Pa_GetErrorText(startErr));
        Pa_CloseStream(stream);
        Pa_Terminate();
        VoiceStream::g_state.running = false;
        return;
    }

    LOG_DEBUG("VoiceStream", "PortAudio stream started successfully.");
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
                LOG_DEBUG("VoiceStream", "Final transcript: " + clean);

                VoiceRuntime::transcript(clean);

                VoiceStream::g_state.partial.clear();
                // Note: Console now uses ConsolePanel with UIInputBox - no direct buffer access
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    LOG_DEBUG("VoiceStream", "Stopping PortAudio stream...");
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    LOG_DEBUG("Voice", "VoiceStream stopped cleanly");
}


// ---------------- Control API ----------------
bool VoiceStream::isRunning() { return g_state.running; }

bool VoiceStream::start(whisper_context* ctx)
{
    // ------------------------------------------------------------
    // Prevent duplicate sessions
    // ------------------------------------------------------------
    if (g_state.running) {
        LOG_DEBUG("VoiceStream", "Start request ignored — already running.");
        return false;
    }

    // ------------------------------------------------------------
    // Join any old thread (if not fully cleaned up)
    // ------------------------------------------------------------
    if (g_state.thread.joinable()) {
        LOG_DEBUG("VoiceStream", "Joining previous voice thread before restart...");
        g_state.thread.join();
    }

    g_silenceThreshold=VoiceRuntime::config["voice"].value("silence_threshold",0.02);
    g_silenceTimeoutMs=VoiceRuntime::config["voice"].value("silence_timeout_ms",1200);
    g_whisperLanguage=VoiceRuntime::config["whisper"].value("language",std::string("en"));
    g_whisperMaxTokens=VoiceRuntime::config["whisper"].value("max_tokens",128);
    g_state.inputDeviceIndex=VoiceRuntime::config["voice"].value("input_device_index",-1);
    // ------------------------------------------------------------
    // Calibrate silence threshold before recording
    // ------------------------------------------------------------
    LOG_DEBUG("VoiceStream", "Calibrating ambient noise threshold...");
    VoiceStream::calibrateSilence();
    LOG_DEBUG("Voice", "Voice calibration complete");

    // ------------------------------------------------------------
    // Launch new listening thread
    // ------------------------------------------------------------
    g_state.running = true;
    g_state.thread = std::thread([ctx]() {
        LOG_DEBUG("VoiceStream", "Voice thread started.");
        try { run(ctx); } catch(const std::exception& error) { VoiceRuntime::reportError(error.what()); }
        g_state.running=false;
        LOG_DEBUG("VoiceStream", "Voice thread ended.");
    });

    LOG_DEBUG("Voice", "VoiceStream started");
    return true;
}


void VoiceStream::stop() {
    if (!g_state.running) {
        LOG_DEBUG("VoiceStream", "Voice stream already stopped.");
        if (g_state.thread.joinable())
            g_state.thread.join();
        return;
    }

    LOG_DEBUG("VoiceStream", "Stopping voice stream...");
    g_state.running = false;

    if (g_state.thread.joinable()) {
        g_state.thread.join();
        LOG_DEBUG("VoiceStream", "Voice thread joined successfully.");
    }

    LOG_DEBUG("VoiceStream", "Voice listening stopped cleanly.");
}

// ---------------- Ambient Calibration ----------------
void VoiceStream::calibrateSilence() {
    LOG_DEBUG("VoiceStream", "Starting ambient calibration...");
    if (Pa_Initialize() != paNoError) {
        LOG_ERROR("VoiceStream", "Failed to initialize PortAudio for calibration.");
        return;
    }

    int deviceIndex = Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        LOG_ERROR("VoiceStream", "No input device available for calibration.");
        Pa_Terminate();
        return;
    }

    const PaDeviceInfo* devInfo = Pa_GetDeviceInfo(deviceIndex);
    PaStreamParameters inputParams;
    inputParams.device = deviceIndex;
    inputParams.channelCount = 1;
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = devInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    std::vector<float> buffer;
    PaStream* stream;
    Pa_OpenStream(&stream, &inputParams, nullptr, 16000, 512, paNoFlag, nullptr, nullptr);
    Pa_StartStream(stream);

    LOG_DEBUG("VoiceStream", "Measuring ambient noise for 2 seconds...");
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
        float chunk[512];
        if (Pa_ReadStream(stream, chunk, 512) == paNoError)
            buffer.insert(buffer.end(), chunk, chunk + 512);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    if (buffer.empty()) {
        LOG_ERROR("VoiceStream", "No audio captured during calibration.");
        return;
    }

    double energy = 0.0;
    for (float s : buffer) energy += s * s;
    double rms = std::sqrt(energy / buffer.size());

    double newThreshold = rms * 1.8;
    g_silenceThreshold = (g_silenceThreshold * 0.5) + (newThreshold * 0.5);

    LOG_DEBUG("Voice", "Voice calibration");
    LOG_DEBUG("VoiceStream",
        "Ambient RMS=" + std::to_string(rms) +
        " → new silence threshold=" + std::to_string(g_silenceThreshold));
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
            if (!isSilence(pcm)) lastSpeechTime = std::chrono::steady_clock::now();

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
                    for (int i = 0; i < n; i++)
                        transcript += whisper_full_get_segment_text(ctx, i);
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
