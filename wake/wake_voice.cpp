#include "pv_porcupine.h"
#include "logger.hpp"
#include <vector>
#include <mutex>
#include <string>
#include <thread>
#include <atomic>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "nlp/nlp.hpp"
#include "commands/commands_core.hpp"
#include "voice/voice_speak.hpp"
#include "popup_ui/popup_ui.hpp"
#include "ai/ai.hpp"
#include "voice/voice.hpp"
#include "bootstrap/bootstrap_config.hpp"

// ==== PortAudio ====
#include <portaudio.h>

// -----------------------------------------------------------------------------
// Fallback definitions for older Porcupine SDK versions
// -----------------------------------------------------------------------------
#ifndef pv_porcupine_sample_rate
    #define pv_porcupine_sample_rate() (16000)   // Porcupine uses 16 kHz PCM
#endif

#ifndef pv_porcupine_frame_length
    #define pv_porcupine_frame_length() (512)    // 512 samples per frame typical
#endif

namespace Voice {

static pv_porcupine_t* g_porcupine = nullptr;
static std::mutex g_mutex;

bool initWakeWord(const std::string& accessKey,
                  const std::string& modelPath,
                  const std::string& keywordPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    LOG_DEBUG("Porcupine", "CWD: " + std::filesystem::current_path().string());
    LOG_DEBUG("Porcupine", "Initializing wake-word engine...");
    LOG_DEBUG("Porcupine", "Access key length: " + std::to_string(accessKey.size()));
    LOG_DEBUG("Porcupine", "Model path: " + modelPath);
    LOG_DEBUG("Porcupine", "Keyword path: " + keywordPath);

    if (!std::filesystem::exists(modelPath)) {
        LOG_ERROR("Porcupine", "Model file missing at: " + modelPath);
        return false;
    }
    if (!std::filesystem::exists(keywordPath)) {
        LOG_ERROR("Porcupine", "Keyword file missing at: " + keywordPath);
        return false;
    }
    if (g_porcupine) {
        LOG_DEBUG("Porcupine", "Wake-word engine already initialized, skipping re-init.");
        return true;
    }

    const char* keywordPaths[] = { keywordPath.c_str() };
    const float sensitivities[] = { 0.5f };

    LOG_DEBUG("Porcupine", "Calling pv_porcupine_init...");
    pv_status_t status = pv_porcupine_init(
        accessKey.c_str(),
        modelPath.c_str(),
        1,
        keywordPaths,
        sensitivities,
        &g_porcupine
    );
    LOG_DEBUG("Porcupine", "pv_porcupine_init returned status: " + std::to_string(status));

    if (status != PV_STATUS_SUCCESS) {
        LOG_ERROR("Porcupine", "Failed to initialize wake-word engine. Status: " + std::to_string(status));
        g_porcupine = nullptr;
        return false;
    }

    LOG_DEBUG("Porcupine", "Wake-word engine initialized successfully!");
    return true;
}

bool detectWakeWordFrame(const int16_t* pcm) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_porcupine) return false;

    int32_t keywordIndex = -1;
    pv_status_t status = pv_porcupine_process(g_porcupine, pcm, &keywordIndex);
    if (status != PV_STATUS_SUCCESS) {
        LOG_ERROR("Porcupine", "Processing error.");
        return false;
    }
    return (keywordIndex >= 0);
}

void shutdownWakeWord() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_porcupine) {
        pv_porcupine_delete(g_porcupine);
        g_porcupine = nullptr;
        LOG_TRACE("Porcupine", "Wakeword engine shut down.");
    }
}

} // namespace Voice


// =========================================================
// WakeVoice high-level startup + listening loop
// =========================================================
namespace {
    std::atomic<bool> g_running{false};
    std::thread g_thread;
    static bool g_listening = false;
}

namespace WakeVoice {

static bool startPortAudio(PaStream** outStream, double desiredRate, int channels) {
    PaError err = Pa_Initialize();
    if (err != paNoError && err != paNotInitialized) {
        LOG_ERROR("WakeVoice", std::string("Pa_Initialize failed: ") + Pa_GetErrorText(err));
        return false;
    }

    PaStreamParameters in{};
    in.device = Pa_GetDefaultInputDevice();
    if (in.device == paNoDevice) {
        LOG_ERROR("WakeVoice", "No default input device.");
        return false;
    }
    in.channelCount = channels;
    in.sampleFormat = paInt16;
    in.suggestedLatency = Pa_GetDeviceInfo(in.device)->defaultLowInputLatency;
    in.hostApiSpecificStreamInfo = nullptr;

    PaError openErr = Pa_OpenStream(
        outStream,
        &in,
        nullptr,
        desiredRate,
        pv_porcupine_frame_length(),
        paNoFlag,
        nullptr,
        nullptr
    );
    if (openErr != paNoError) {
        LOG_ERROR("WakeVoice", std::string("Pa_OpenStream failed: ") + Pa_GetErrorText(openErr));
        return false;
    }

    PaError startErr = Pa_StartStream(*outStream);
    if (startErr != paNoError) {
        LOG_ERROR("WakeVoice", std::string("Pa_StartStream failed: ") + Pa_GetErrorText(startErr));
        Pa_CloseStream(*outStream);
        *outStream = nullptr;
        return false;
    }

    LOG_DEBUG("WakeVoice", "PortAudio input stream started.");
    return true;
}

static void stopPortAudio(PaStream* stream) {
    if (stream) {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
    }
    Pa_Terminate();
}

void start(ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp)
{
    if (g_running.load()) {
        LOG_DEBUG("WakeVoice", "Already running.");
        return;
    }

    LOG_DEBUG("WakeVoice", "Starting wake-word listener...");

    const std::string accessKey  = "l24x+8ku2pUsbZKcEyICgbx3Aj/15JHoqGj1TQr+JHcyCXA2RSV2LA==";
    const std::string modelPath  = "D:/G.R.I.M/external/porcupine/lib/common/porcupine_params.pv";
    const std::string keywordPath= "D:/G.R.I.M/resources/wakeword/grim.ppn";

    if (!Voice::initWakeWord(accessKey, modelPath, keywordPath)) {
        LOG_ERROR("WakeVoice", "Failed to initialize wake-word engine.");
        return;
    }

    g_running = true;
    g_thread = std::thread([history, &timers, &longTermMemory, &nlp]() {
        const int frameLen = pv_porcupine_frame_length();
        const int sampleRate = pv_porcupine_sample_rate();
        std::vector<int16_t> frame(frameLen);

        PaStream* stream = nullptr;
        if (!startPortAudio(&stream, sampleRate, 1)) {
            LOG_ERROR("WakeVoice", "Audio init failed; stopping.");
            g_running = false;
            return;
        }

        LOG_PHASE("WakeVoice", "Listening for wake word...");

        while (g_running.load()) {
            PaError r = Pa_ReadStream(stream, frame.data(), frameLen);
            if (r != paNoError) {
                LOG_ERROR("WakeVoice", std::string("Pa_ReadStream error: ") + Pa_GetErrorText(r));
                break;
            }

            if (Voice::detectWakeWordFrame(frame.data())) {
                LOG_TRACE("WakeVoice", "Wake word detected!");
                notifyPopupActivity();

                if (g_listening) continue;
                g_listening = true;

                Voice::speak("Yes?", "wake");

                // Capture and process the user's spoken command (same as WakeKey)
                std::string transcript = Voice::runVoiceDemo(aiConfig, longTermMemory);
                LOG_DEBUG("WakeVoice", "Captured voice transcript: " + transcript);

                if (!transcript.empty()) {
                    handleCommand(transcript); // unified command pipeline
                }

                g_listening = false;

                // Optional: break for one-shot mode
                // break;
            }
        }

        stopPortAudio(stream);
        LOG_DEBUG("WakeVoice", "Listening thread exiting.");
    });

    LOG_DEBUG("WakeVoice", "Wake-word system initialized and listening.");
}

void stop() {
    if (!g_running.load()) return;
    g_running = false;
    if (g_thread.joinable()) g_thread.join();
    Voice::shutdownWakeWord();
    LOG_DEBUG("WakeVoice", "Stopped.");
}

bool isRunning() { return g_running.load(); }

} // namespace WakeVoice
