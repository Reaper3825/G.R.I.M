#pragma once
#include <string>

/// Holds detected system capabilities
struct SystemInfo {
    // OS / platform
    std::string osName;
    std::string arch;

    // CPU
    int cpuCores = 0;

    // Memory
    long ramMB = 0;

    // GPU
    bool hasGPU = false;
    bool hasCUDA = false;
    bool hasROCm = false;
    bool hasMetal = false;
    int  gpuCount = 0;
    std::string gpuName;
    long gpuVRAM_MB = 0;       // new: VRAM in MB
    std::string gpuDriver;     // new: driver version or provider

    // AI model suggestion
    std::string suggestedModel;

    // Voice backends
    bool hasSAPI  = false;  // Windows Speech API
    bool hasSay   = false;  // macOS "say" command
    bool hasPiper = false;  // Linux Piper TTS
};


/// Detects hardware/system capabilities (CPU, GPU, RAM, OS, voice backends).
SystemInfo detectSystem();

/// Prints the system info to logs/console.
void logSystemInfo(const SystemInfo& info);

/// Decide the best Whisper model name to use based on system resources.
std::string chooseWhisperModel(const SystemInfo& info);
