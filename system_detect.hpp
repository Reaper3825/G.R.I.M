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

    // AI model suggestion
    std::string suggestedModel;
};

/// Detects hardware/system capabilities (CPU, GPU, RAM, OS).
SystemInfo detectSystem();

/// Prints the system info to logs/console.
void logSystemInfo(const SystemInfo& info);

/// Decide the best Whisper model name to use based on system resources.
std::string chooseWhisperModel(const SystemInfo& info);
