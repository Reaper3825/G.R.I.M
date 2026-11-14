#include "audio_core.hpp"
#include "runtime_state.hpp"
#include <atomic>
#include "plugins/plugin_logging.hpp"
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

static std::atomic<bool> g_initialized{false};
static std::mutex initMutex;

// ---------------- Init / Shutdown ----------------
bool init() {
    std::lock_guard<std::mutex> lock(initMutex);
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
std::mutex playbackMutex;
std::atomic<bool> playing{false}, stopRequested{false};
std::atomic<float> volume{1.0f};
}
bool playWav(const std::string& path) {
    std::lock_guard<std::mutex> lock(playbackMutex);
    if (VoiceRuntime::cancel.load()) return false;
    std::vector<int16_t> pcm;
    int sampleRate=0, channels=0;
    if (!readWav(path,pcm,sampleRate,channels) || channels<=0 || sampleRate<=0) return false;
    if (!init()) return false;
    for(auto& sample:pcm) sample=static_cast<int16_t>(sample*volume.load());
    PaStream* stream=nullptr;
    if(Pa_OpenDefaultStream(&stream,0,channels,paInt16,sampleRate,paFramesPerBufferUnspecified,nullptr,nullptr)!=paNoError) return false;
    stopRequested=false;
    auto error=Pa_StartStream(stream);
    playing=error==paNoError;
    const size_t frames=pcm.size()/channels;
    for(size_t pos=0; error==paNoError && pos<frames && !stopRequested && !VoiceRuntime::cancel; ) {
        const auto chunk=std::min<size_t>(1024,frames-pos);
        error=Pa_WriteStream(stream,pcm.data()+pos*channels,static_cast<unsigned long>(chunk));
        if(error==paOutputUnderflowed) error=paNoError;
        pos+=chunk;
    }
    Pa_AbortStream(stream); Pa_CloseStream(stream); playing=false;
    return error==paNoError && !stopRequested && !VoiceRuntime::cancel;
}
bool isPlaying() { return playing; }
void stopAll() { stopRequested=true; }
void stopCurrent() { stopAll(); }
void setVolume(float value) { volume=std::clamp(value,0.0f,1.0f); }

} // namespace Audio
