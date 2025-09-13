#include "voice.hpp"
#include "resources.hpp"
#include "ai.hpp"
#include "voice_speak.hpp"
#include "system_detect.hpp"   // OS detection authority
#include "response_manager.hpp" // 🔹 NEW: natural replies

#include <whisper.h>
#include <portaudio.h>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <chrono>
#include <nlohmann/json.hpp>
#include <cpr/cpr.h>   // HTTP for online TTS

namespace fs = std::filesystem;

namespace Voice {

// ---------------- State ----------------
State g_state;

static double g_silenceThreshold = 0.02;
static int g_silenceTimeoutMs = 4000;
static std::string g_ttsUrl = "http://127.0.0.1:8080/tts"; // default

// ---------------- Audio Data ----------------
struct AudioData {
    std::vector<float> buffer;
    bool ready = false;
};

// ============================================================
// PortAudio Helpers
// ============================================================
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

static int playbackCallback(const void* /*input*/,
                            void* output,
                            unsigned long frameCount,
                            const PaStreamCallbackTimeInfo* /*timeInfo*/,
                            PaStreamCallbackFlags /*statusFlags*/,
                            void* userData) {
    AudioData* data = reinterpret_cast<AudioData*>(userData);
    float* out = reinterpret_cast<float*>(output);

    for (unsigned long i = 0; i < frameCount; i++) {
        if (!data->buffer.empty()) {
            *out++ = data->buffer.front();
            data->buffer.erase(data->buffer.begin());
        } else {
            *out++ = 0.0f; // silence
        }
    }
    return data->buffer.empty() ? paComplete : paContinue;
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
// Whisper Initialization
// ============================================================
bool initWhisper(const std::string& modelName, std::string* err) {
    fs::path modelPathFS = fs::path(getResourcePath()) / "models" / "whisper" / ("ggml-" + modelName + ".bin");

    std::cerr << "[DEBUG][Voice] Looking for Whisper model at " << modelPathFS << "\n";

    if (!fs::exists(modelPathFS)) {
        if (err) *err = "Model not found at " + modelPathFS.string();
        return false;
    }

    struct whisper_context_params wparams = whisper_context_default_params();
    g_state.ctx = whisper_init_from_file_with_params(modelPathFS.string().c_str(), wparams);

    if (!g_state.ctx) {
        if (err) *err = "Failed to load Whisper model";
        return false;
    }

    std::cerr << "[DEBUG][Voice] Whisper model loaded OK\n";

    // --- Pull settings from aiConfig["voice"]
    if (aiConfig.contains("voice")) {
        auto& v = aiConfig["voice"];
        g_state.inputDeviceIndex = v.value("input_device_index", -1);
        g_silenceTimeoutMs       = v.value("silence_timeout_ms", 4000);
        g_ttsUrl                 = v.value("tts_url", g_ttsUrl);
        std::cerr << "[DEBUG][Voice] Using voice block: device=" << g_state.inputDeviceIndex
                  << " timeout=" << g_silenceTimeoutMs << "ms url=" << g_ttsUrl << "\n";
    } else {
        std::cerr << "[Voice] No 'voice' block in ai_config.json, using defaults.\n";
        g_state.inputDeviceIndex = aiConfig.value("input_device_index", -1);
        g_silenceTimeoutMs       = aiConfig.value("silence_timeout_ms", 4000);
        g_ttsUrl                 = aiConfig.value("tts_url", g_ttsUrl);
    }

    // --- Whisper sub-block
    if (aiConfig.contains("whisper")) {
        auto& w = aiConfig["whisper"];
        g_state.minSpeechMs  = w.value("min_speech_ms", 500);
        g_state.minSilenceMs = w.value("min_silence_ms", 1200);
        std::cerr << "[DEBUG][Voice] Whisper sub-block: minSpeech=" << g_state.minSpeechMs
                  << " minSilence=" << g_state.minSilenceMs << "\n";
    }

    return true;
}

// ============================================================
// Voice Input (Speech → Text)
// ============================================================
std::string runVoiceDemo(nlohmann::json& longTermMemory) {
    std::cerr << "[DEBUG][Voice] Entering runVoiceDemo()\n";

    if (!g_state.ctx) {
        std::cerr << "[ERROR][Voice] Whisper not initialized!\n";
        return "";
    }

    if (Pa_Initialize() != paNoError) {
        std::cerr << "[ERROR][Voice] Failed to init PortAudio\n";
        return "";
    }

    AudioData data;
    PaStream* stream;

    int deviceIndex = (g_state.inputDeviceIndex >= 0) ? g_state.inputDeviceIndex : Pa_GetDefaultInputDevice();
    if (deviceIndex == paNoDevice) {
        Pa_Terminate();
        std::cerr << "[ERROR][Voice] No input device found\n";
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
        std::cerr << "[ERROR][Voice] Failed to open audio stream\n";
        return "";
    }

    Pa_StartStream(stream);
    std::cerr << "[DEBUG][Voice] " << ResponseManager::get("voice_start") << "\n";

    std::vector<float> rollingBuffer;
    auto lastSpeech = std::chrono::steady_clock::now();
    auto speechStart = lastSpeech;
    bool inSpeech = false;

    while (true) {
        if (data.buffer.size() >= 8000) {
            std::vector<float> chunk(data.buffer.begin(), data.buffer.begin() + 8000);
            data.buffer.erase(data.buffer.begin(), data.buffer.begin() + 8000);

            bool silent = isSilence(chunk);
            std::cerr << "[DEBUG][Voice] Got " << chunk.size() << " samples, silence=" << silent << "\n";

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
                                         std::chrono::steady_clock::now() - lastSpeech)
                                         .count();
                auto msSpeech = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    lastSpeech - speechStart)
                                    .count();

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
        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cerr << "[DEBUG][Voice] Stream stopped\n";

    std::string transcript;
    if (!rollingBuffer.empty()) {
        std::cerr << "[DEBUG][Voice] Running Whisper on " << rollingBuffer.size() << " samples\n";
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
        std::cerr << "[DEBUG][Voice] " << ResponseManager::get("voice_none") << "\n";
    }

    return transcript;
}

// ============================================================
// Voice Output (Text → Speech)
// ============================================================
static bool playPCM(const std::vector<float>& pcm, int sampleRate = 22050) {
    if (Pa_Initialize() != paNoError) {
        std::cerr << "[ERROR][Voice] PortAudio init failed for playback\n";
        return false;
    }

    AudioData data;
    data.buffer = pcm;

    PaStream* stream;
    PaStreamParameters outputParams;
    outputParams.device = Pa_GetDefaultOutputDevice();
    outputParams.channelCount = 1;
    outputParams.sampleFormat = paFloat32;
    outputParams.suggestedLatency = Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    if (Pa_OpenStream(&stream, nullptr, &outputParams, sampleRate, 512, paNoFlag, playbackCallback, &data) != paNoError) {
        Pa_Terminate();
        std::cerr << "[ERROR][Voice] Failed to open playback stream\n";
        return false;
    }

    Pa_StartStream(stream);
    std::cerr << "[DEBUG][Voice] Playing PCM...\n";

    while (Pa_IsStreamActive(stream)) {
        Pa_Sleep(50);
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cerr << "[DEBUG][Voice] Playback finished\n";
    return true;
}

static std::vector<float> fetchOnlineTTS(const std::string& text) {
    std::cerr << "[DEBUG][Voice] Fetching online TTS for: " << text << "\n";
    std::vector<float> pcm;

    auto response = cpr::Post(cpr::Url{g_ttsUrl},
                              cpr::Body{text},
                              cpr::Header{{"Content-Type", "text/plain"}});

    if (response.status_code == 200) {
        std::string body = response.text;
        const float* samples = reinterpret_cast<const float*>(body.data());
        pcm.assign(samples, samples + (body.size() / sizeof(float)));
        std::cerr << "[DEBUG][Voice] TTS response OK, samples=" << pcm.size() << "\n";
    } else {
        std::cerr << "[ERROR][Voice] TTS HTTP " << response.status_code << "\n";
    }
    return pcm;
}

// Offline fallback → system_detect authority
static std::vector<float> generateOfflineTTS(const std::string& text) {
    auto sysInfo = detectSystem();
    std::cerr << "[Voice] " << ResponseManager::get("voice_fallback") 
              << " (" << sysInfo.osName << ")\n";

    std::string localEngine = "en_US-amy-medium.onnx";
    if (aiConfig.contains("voice") && aiConfig["voice"].is_object()) {
        localEngine = aiConfig["voice"].value("local_engine", localEngine);
    }

    if (speakLocal(text, localEngine)) {
        if (sysInfo.hasSAPI)  std::cerr << "[Voice] Local speech via Windows SAPI\n";
        if (sysInfo.hasSay)   std::cerr << "[Voice] Local speech via macOS say\n";
        if (sysInfo.hasPiper) std::cerr << "[Voice] Local speech via Linux Piper\n";
    } else {
        std::cerr << "[Voice] Local speech failed.\n";
    }

    return {}; // playback already handled
}

bool speakText(const std::string& text, bool preferOnline) {
    auto sysInfo = detectSystem();
    std::string mode = "hybrid";
    if (aiConfig.contains("voice") && aiConfig["voice"].is_object()) {
        mode = aiConfig["voice"].value("mode", "hybrid");
    }

    std::cerr << "[DEBUG][Voice] speakText called, text=\"" << text 
              << "\" preferOnline=" << preferOnline 
              << " mode=" << mode << "\n";

    // --- Windows: force SAPI unless explicitly "cloud"
    if (sysInfo.hasSAPI && mode != "cloud") {
        generateOfflineTTS(text);
        return true;
    }

    // --- Other OS / explicit cloud
    std::vector<float> pcm;
    if (preferOnline) {
        pcm = fetchOnlineTTS(text);
    }

    if (pcm.empty()) {
        pcm = generateOfflineTTS(text);
    }

    if (!pcm.empty()) {
        return playPCM(pcm);
    } else {
        return true; // assume local handled it
    }
}

} // namespace Voice
