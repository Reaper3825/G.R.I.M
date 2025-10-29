#include "audio_core.hpp"
#include "logger.hpp"
#include <portaudio.h>
#include <vector>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cmath>

// =============================================================
// WAV loader 
// =============================================================
namespace {
struct WavHeader {
    char riff[4];
    uint32_t overall_size;
    char wave[4];
    char fmt_chunk_marker[4];
    uint32_t length_of_fmt;
    uint16_t format_type;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byterate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data_chunk_header[4];
    uint32_t data_size;
};

bool readWav(const std::string& path, std::vector<int16_t>& data,
             int& sampleRate, int& channels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    WavHeader hdr{};
    file.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (std::string(hdr.riff, 4) != "RIFF" || std::string(hdr.wave, 4) != "WAVE")
        return false;

    sampleRate = hdr.sample_rate;
    channels   = hdr.channels;
    data.resize(hdr.data_size / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(data.data()), hdr.data_size);
    return true;
}
} // namespace

// =============================================================
// AudioCore implementation
// =============================================================
namespace Audio {

static bool g_initialized = false;

// ---------------- Init / Shutdown ----------------
bool init() {
    if (g_initialized) return true;
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        LOG_ERROR("AudioCore", std::string("PortAudio init failed: ") + Pa_GetErrorText(err));
        return false;
    }
    g_initialized = true;
    LOG_DEBUG("AudioCore", "PortAudio initialized successfully.");
    return true;
}

void shutdown() {
    if (!g_initialized) return;
    Pa_Terminate();
    g_initialized = false;
    LOG_DEBUG("AudioCore", "PortAudio shutdown complete.");
}

// ---------------- Device Management ----------------
std::vector<DeviceInfo> listDevices() {
    std::vector<DeviceInfo> result;
    if (!g_initialized) init();

    int num = Pa_GetDeviceCount();
    if (num < 0) {
        LOG_ERROR("AudioCore", "Failed to get device count.");
        return result;
    }

    int defIn = Pa_GetDefaultInputDevice();
    int defOut = Pa_GetDefaultOutputDevice();

    for (int i = 0; i < num; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) continue;
        DeviceInfo d;
        d.name = info->name ? info->name : "Unknown";
        d.isInput = (info->maxInputChannels > 0);
        d.isOutput = (info->maxOutputChannels > 0);
        d.isDefault = (i == defIn) || (i == defOut);
        result.push_back(d);
    }
    return result;
}

std::string getDefaultOutput() {
    if (!g_initialized) init();
    int idx = Pa_GetDefaultOutputDevice();
    if (idx == paNoDevice) return "Unknown";
    const PaDeviceInfo* info = Pa_GetDeviceInfo(idx);
    return info && info->name ? info->name : "Unknown";
}

std::string getDefaultInput() {
    if (!g_initialized) init();
    int idx = Pa_GetDefaultInputDevice();
    if (idx == paNoDevice) return "Unknown";
    const PaDeviceInfo* info = Pa_GetDeviceInfo(idx);
    return info && info->name ? info->name : "Unknown";
}

// =============================================================
// Internal playback tracker
// =============================================================
namespace {
struct ActiveStream {
    PaStream* stream = nullptr;
    std::vector<int16_t> buffer;
    int channels = 0;
    int sampleRate = 0;
    bool playing = false;
};

static std::vector<std::unique_ptr<ActiveStream>> g_active;
static std::mutex g_audioMutex;
static float g_masterVolume = 1.0f;

static void cleanupFinished() {
    std::lock_guard<std::mutex> lock(g_audioMutex);
    g_active.erase(std::remove_if(g_active.begin(), g_active.end(),
        [](const std::unique_ptr<ActiveStream>& s) {
            if (!s || !s->stream) return true;
            return !s->playing || Pa_IsStreamActive(s->stream) == 0;
        }),
        g_active.end());
}
} // namespace

// =============================================================
// WAV Playback
// =============================================================
bool playWav(const std::string& path) {
    if (!g_initialized && !init()) return false;

    std::vector<int16_t> pcm;
    int sampleRate = 0, channels = 0;
    if (!readWav(path, pcm, sampleRate, channels)) {
        LOG_ERROR("AudioCore", "Invalid WAV: " + path);
        return false;
    }

    // Apply master volume scaling
    for (auto& s : pcm)
        s = static_cast<int16_t>(s * g_masterVolume);

    auto active = std::make_unique<ActiveStream>();
    active->channels = channels;
    active->sampleRate = sampleRate;
    active->playing = true;  // ? Set BEFORE opening stream

    PaError err = Pa_OpenDefaultStream(&active->stream,
                                       0, channels, paInt16,
                                       sampleRate, paFramesPerBufferUnspecified,
                                       nullptr, nullptr);
    if (err != paNoError || !active->stream) {
        LOG_ERROR("AudioCore", "Stream open failed: " + std::string(Pa_GetErrorText(err)));
        return false;
    }

    // ? Add to active list BEFORE playing (so isPlaying() works)
    {
        std::lock_guard<std::mutex> lock(g_audioMutex);
        g_active.push_back(std::move(active));
    }

    // ? Get the stream pointer (active was moved)
    PaStream* stream = g_active.back()->stream;

    Pa_StartStream(stream);
    size_t frames = pcm.size() / channels;
    const int16_t* data = pcm.data();
    size_t pos = 0;

    while (pos < frames) {
        size_t chunk = std::min<size_t>(1024, frames - pos);
        Pa_WriteStream(stream, data + pos * channels, (unsigned long)chunk);
        pos += chunk;
    }

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    
    // ? Mark as finished AFTER playback completes
    {
        std::lock_guard<std::mutex> lock(g_audioMutex);
        if (!g_active.empty()) {
            g_active.back()->playing = false;
            g_active.back()->stream = nullptr;
        }
    }

    cleanupFinished();
    return true;
}

// =============================================================
// Playback Control / Queries
// =============================================================
bool isPlaying() {
    std::lock_guard<std::mutex> lock(g_audioMutex);
    for (auto& s : g_active) {
        if (s && s->playing && Pa_IsStreamActive(s->stream) == 1)
            return true;
    }
    return false;
}

void stopAll() {
    std::lock_guard<std::mutex> lock(g_audioMutex);
    for (auto& s : g_active) {
        if (s && s->stream) {
            Pa_StopStream(s->stream);
            Pa_CloseStream(s->stream);
            s->playing = false;
        }
    }
    g_active.clear();
    LOG_DEBUG("AudioCore", "All active streams stopped.");
}

void stopCurrent() {
    std::lock_guard<std::mutex> lock(g_audioMutex);
    if (!g_active.empty()) {
        auto& s = g_active.back();
        if (s && s->stream) {
            Pa_StopStream(s->stream);
            Pa_CloseStream(s->stream);
            s->playing = false;
            g_active.pop_back();
            LOG_DEBUG("AudioCore", "Stopped current stream.");
        }
    }
}

void setVolume(float volume) {
    g_masterVolume = std::clamp(volume, 0.0f, 1.0f);
    LOG_DEBUG("AudioCore", "Master volume set to " + std::to_string(g_masterVolume));
}

} // namespace Audio
