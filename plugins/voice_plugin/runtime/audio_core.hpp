#pragma once
#include <string>
#include <vector>

// =============================================================
// GRIM AudioCore — PortAudio wrapper for playback, devices, etc.
// =============================================================
namespace Audio {

struct DeviceInfo {
    std::string name;
    bool isDefault = false;
    bool isInput = false;
    bool isOutput = false;
};

// -------------------------------------------------------------
// Initialization / teardown
// -------------------------------------------------------------
bool init();
void shutdown();

// -------------------------------------------------------------
// Device management
// -------------------------------------------------------------
std::vector<DeviceInfo> listDevices();
std::string getDefaultOutput();
std::string getDefaultInput();

// -------------------------------------------------------------
// Playback
// -------------------------------------------------------------
bool playWav(const std::string& path);

// -------------------------------------------------------------
// Playback control / tracking
// -------------------------------------------------------------
bool isPlaying();             // true if any stream is active
void stopAll();               // stop all active streams
void stopCurrent();           // stop last played stream
void setVolume(float volume); // 0.0–1.0 master volume (applied to future playbacks)

} // namespace Audio
