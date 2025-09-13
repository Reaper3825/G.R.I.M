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
#endif

// CUDA headers (only if compiled with cuBLAS)
#ifdef GGML_USE_CUBLAS
#include <cuda_runtime.h>
#endif

SystemInfo detectSystem() {
    SystemInfo info;

    // --- OS Detection ---
    #ifdef _WIN32
    info.osName = "Windows";
    #elif __APPLE__
    info.osName = "macOS";
    #elif __linux__
    info.osName = "Linux";
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
    // whisper.cpp automatically uses Metal if available,
    // so we can just mark it here
    info.hasGPU = true;
    info.hasMetal = true;
    #endif

    // TODO: ROCm detection for AMD

    // --- Suggested Model ---
    info.suggestedModel = chooseWhisperModel(info);

    return info;
}

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

    std::cout << "Suggested Whisper model: " << info.suggestedModel << "\n";
    std::cout << "-------------------------------\n";
}

std::string chooseWhisperModel(const SystemInfo& info) {
    // Decide based on RAM and GPU presence
    if (info.hasGPU && info.ramMB > 16000) {
        return "large-v3";   // Big systems with GPU + lots of RAM
    }
    if (info.hasGPU && info.ramMB > 8000) {
        return "medium";
    }
    if (info.ramMB > 4000) {
        return "small";
    }
    return "base.en"; // Fallback for low-power systems
}
