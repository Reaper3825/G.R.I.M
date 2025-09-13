#include "system_detect.hpp"
#include <iostream>
#include <thread>
#include <cstdlib>

// Platform headers
#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif __linux__
#include <sys/utsname.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// CUDA headers (only if compiled with cuBLAS)
#ifdef GGML_USE_CUBLAS
#include <cuda_runtime.h>
#endif

// =========================================================
// Dependency helpers (Linux Piper detection)
// =========================================================
#ifdef __linux__
static bool commandExists(const char* cmd) {
    std::string check = "which " + std::string(cmd) + " > /dev/null 2>&1";
    return (system(check.c_str()) == 0);
}

bool ensurePiperInstalled() {
    if (commandExists("piper")) {
        return true; // already installed
    }

    std::cerr << "[SystemDetect] Piper not found. Attempting to install...\n";

    if (commandExists("apt-get")) {
        return system("sudo apt-get update && sudo apt-get install -y piper-tts") == 0;
    }
    if (commandExists("dnf")) {
        return system("sudo dnf install -y piper-tts") == 0;
    }
    if (commandExists("pacman")) {
        return system("sudo pacman -S --noconfirm piper-tts") == 0;
    }

    std::cerr << "[SystemDetect] Could not auto-install Piper.\n"
              << "Please install manually: https://github.com/rhasspy/piper\n";
    return false;
}
#endif

// =========================================================
// System detection
// =========================================================
SystemInfo detectSystem() {
    SystemInfo info;

    // --- OS Detection ---
    #ifdef _WIN32
    info.osName = "Windows";
    info.hasSAPI = true; // Always available
    #elif __APPLE__
    info.osName = "macOS";
    info.hasSay = true; // say command always available
    #elif __linux__
    info.osName = "Linux";
    // Piper check
    info.hasPiper = ensurePiperInstalled();
    #else
    info.osName = "Unknown";
    #endif

    // --- Architecture ---
    #if defined(__x86_64__) || defined(_M_X64)
    info.arch = "x86_64";
    #elif defined(__aarch64__)
    info.arch = "ARM64";
    #elif defined(__arm__)
    info.arch = "ARM";
    #else
    info.arch = "Unknown";
    #endif

    // --- CPU ---
    info.cpuCores = std::thread::hardware_concurrency();

    // --- RAM ---
    #ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    info.ramMB = status.ullTotalPhys / (1024 * 1024);
    #elif __APPLE__
    int64_t mem;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    info.ramMB = mem / (1024 * 1024);
    #elif __linux__
    struct sysinfo sys;
    if (sysinfo(&sys) == 0) {
        info.ramMB = sys.totalram / (1024 * 1024);
    }
    #endif

    // --- GPU (CUDA) ---
    #ifdef GGML_USE_CUBLAS
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        info.hasGPU = true;
        info.hasCUDA = true;
        info.gpuCount = deviceCount;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        info.gpuName = prop.name;
    }
    #endif

    // --- Metal (macOS) ---
    #ifdef __APPLE__
    info.hasGPU = true;
    info.hasMetal = true;
    #endif

    // --- Suggested Whisper Model ---
    info.suggestedModel = chooseWhisperModel(info);

    return info;
}


// =========================================================
// Logging
// =========================================================
void logSystemInfo(const SystemInfo& info) {
    std::cout << "---- GRIM System Detection ----\n";
    std::cout << "OS: " << info.osName << " (" << info.arch << ")\n";
    std::cout << "CPU cores: " << info.cpuCores << "\n";
    std::cout << "RAM: " << info.ramMB << " MB\n";

    if (info.hasGPU) {
        std::cout << "GPU detected: " << info.gpuName 
                  << " (" << info.gpuCount << " device(s))\n";
        if (info.hasCUDA) std::cout << "CUDA supported.\n";
        if (info.hasMetal) std::cout << "Metal supported.\n";
        if (info.hasROCm) std::cout << "ROCm supported.\n";
    } else {
        std::cout << "No GPU detected, running CPU-only mode.\n";
    }

    // Voice engines
    std::cout << "Voice backends:\n";
    std::cout << "  Windows SAPI: " << (info.hasSAPI ? "Yes" : "No") << "\n";
    std::cout << "  macOS say:   " << (info.hasSay ? "Yes" : "No") << "\n";
    std::cout << "  Linux Piper: " << (info.hasPiper ? "Yes" : "No") << "\n";

    std::cout << "Suggested Whisper model: " << info.suggestedModel << "\n";
    std::cout << "-------------------------------\n";
}


// =========================================================
// Whisper model chooser
// =========================================================
std::string chooseWhisperModel(const SystemInfo& info) {
    if (info.hasGPU && info.ramMB > 16000) {
        return "large-v3";
    }
    if (info.hasGPU && info.ramMB > 8000) {
        return "medium";
    }
    if (info.ramMB > 4000) {
        return "small";
    }
    return "base.en";
}
