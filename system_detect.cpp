#include "system_detect.hpp"
#include "logger.hpp"

#include <cstdlib>
#include <thread>
#include <SFML/Audio.hpp>
#include <iostream>

// =========================================================
// Platform headers
// =========================================================
#ifdef _WIN32
    #include <windows.h>
    #include <mmdeviceapi.h>
    #include <functiondiscoverykeys_devpkey.h>
    #include <comdef.h>
    #include <Wbemidl.h>
    #include <dxgi.h>
    #pragma comment(lib, "ole32.lib")
    #pragma comment(lib, "wbemuuid.lib")
    #pragma comment(lib, "dxgi.lib")
#elif __APPLE__
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <sys/utsname.h>
#elif __linux__
    #include <sys/utsname.h>
    #include <sys/sysinfo.h>
    #include <unistd.h>
#endif

#ifdef GGML_USE_CUBLAS
    #include <cuda_runtime.h>
#endif

// =========================================================
// Windows helpers
// =========================================================
#ifdef _WIN32
static std::string wideToUtf8(const BSTR& wstr) {
    if (!wstr) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (size_needed <= 0) return "";
    std::string str(size_needed - 1, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &str[0], size_needed, NULL, NULL);
    return str;
}

// 🔹 Forward declaration so detectSystem() can see it
static bool detectWindowsGPU(SystemInfo& info);
#endif

// =========================================================
// Linux helpers
// =========================================================
#ifdef __linux__
static bool commandExists(const char* cmd) {
    std::string check = "which " + std::string(cmd) + " > /dev/null 2>&1";
    return (system(check.c_str()) == 0);
}

bool ensurePiperInstalled() {
    if (commandExists("piper")) return true;
    LOG_ERROR("SystemDetect", "Piper not found. Please install manually.");
    return false;
}
#endif

// =========================================================
// Windows GPU detection (DXGI + WMI)
// =========================================================
#ifdef _WIN32
static bool detectWindowsGPU(SystemInfo& info) {
    IDXGIFactory* pFactory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory))) {
        return false;
    }

    IDXGIAdapter* pAdapter = nullptr;
    int index = 0;
    int gpuCount = 0;
    long gpuVRAM = 0;
    std::string gpuName;

    while (pFactory->EnumAdapters(index, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC desc;
        if (SUCCEEDED(pAdapter->GetDesc(&desc))) {
            std::wstring ws(desc.Description);
            std::string name(ws.begin(), ws.end());

            // Filter NVIDIA adapters
            if (name.find("NVIDIA") != std::string::npos) {
                gpuCount++;
                gpuName = name;
                gpuVRAM = static_cast<long>(desc.DedicatedVideoMemory / (1024 * 1024));
            }
        }
        pAdapter->Release();
        index++;
    }
    pFactory->Release();

    if (gpuCount == 0) return false;

    // Use WMI for driver version
    HRESULT hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (SUCCEEDED(hres) || hres == RPC_E_CHANGED_MODE) {
        IWbemLocator* pLoc = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER,
                                       IID_IWbemLocator, (LPVOID*)&pLoc))) {
            IWbemServices* pSvc = nullptr;
            if (SUCCEEDED(pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"),
                                              NULL, NULL, 0, NULL, 0, 0, &pSvc))) {
                CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, NULL,
                                  RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
                                  NULL, EOAC_NONE);

                IEnumWbemClassObject* pEnumerator = nullptr;
                if (SUCCEEDED(pSvc->ExecQuery(bstr_t("WQL"),
                                              bstr_t("SELECT Name, DriverVersion FROM Win32_VideoController"),
                                              WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                                              NULL, &pEnumerator))) {
                    IWbemClassObject* pclsObj = nullptr;
                    ULONG uReturn = 0;
                    while (pEnumerator && pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn) == S_OK) {
                        VARIANT vtProp;
                        if (SUCCEEDED(pclsObj->Get(L"Name", 0, &vtProp, 0, 0))) {
                            if (vtProp.vt == VT_BSTR) {
                                std::string name = wideToUtf8(vtProp.bstrVal);
                                if (name.find("NVIDIA") != std::string::npos) {
                                    VARIANT vtDriver;
                                    if (SUCCEEDED(pclsObj->Get(L"DriverVersion", 0, &vtDriver, 0, 0))) {
                                        if (vtDriver.vt == VT_BSTR) {
                                            info.gpuDriver = wideToUtf8(vtDriver.bstrVal);
                                        }
                                        VariantClear(&vtDriver);
                                    }
                                }
                            }
                            VariantClear(&vtProp);
                        }
                        pclsObj->Release();
                    }
                    if (pEnumerator) pEnumerator->Release();
                }
                pSvc->Release();
            }
            pLoc->Release();
        }
        CoUninitialize();
    }

    info.hasGPU = true;
    info.gpuCount = gpuCount;
    info.gpuName = gpuName;
    info.gpuVRAM_MB = gpuVRAM;
    return true;
}
#endif

// =========================================================
// Output device selection
// =========================================================
static void selectOutputDevice(SystemInfo& info) {
    auto devices = sf::SoundBufferRecorder::getAvailableDevices();

    if (devices.empty()) {
        LOG_ERROR("SystemDetect", "No output devices detected");
        return;
    }

    std::cout << "Available playback devices:\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i] << "\n";
    }

    int choice = -1;
    std::cout << "Select output device by number: ";
    std::cin >> choice;

    if (choice >= 0 && choice < (int)devices.size()) {
        info.outputDevice = devices[choice];
        LOG_PHASE("Output device chosen", true);
        LOG_DEBUG("SystemDetect", "User selected: " + info.outputDevice);
    } else {
        info.outputDevice = sf::SoundBufferRecorder::getDefaultDevice();
        LOG_PHASE("Output device defaulted", true);
    }

    std::cout << "[Audio] You selected: " << info.outputDevice << "\n";
}

// =========================================================
// Main detection entry
// =========================================================
SystemInfo detectSystem() {
    SystemInfo info;

    // OS + voice backends
#ifdef _WIN32
    info.osName = "Windows";
    info.hasSAPI = true;
#elif __APPLE__
    info.osName = "macOS";
    info.hasSay = true;
#elif __linux__
    info.osName = "Linux";
    info.hasPiper = ensurePiperInstalled();
#else
    info.osName = "Unknown";
#endif

    // Architecture
#if defined(__x86_64__) || defined(_M_X64)
    info.arch = "x86_64";
#elif defined(__aarch64__)
    info.arch = "ARM64";
#elif defined(__arm__)
    info.arch = "ARM";
#else
    info.arch = "Unknown";
#endif

    // CPU
    info.cpuCores = std::thread::hardware_concurrency();

    // RAM
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

    // GPU via CUDA
#ifdef GGML_USE_CUBLAS
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        info.hasGPU = true;
        info.hasCUDA = true;
        info.gpuCount = deviceCount;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        info.gpuName = prop.name;
        info.gpuVRAM_MB = prop.totalGlobalMem / (1024 * 1024);
    }
#endif

    // Windows fallback GPU detection
#ifdef _WIN32
    if (!info.hasGPU) {
        detectWindowsGPU(info);
    }
#endif

    // macOS Metal
#ifdef __APPLE__
    info.hasGPU = true;
    info.hasMetal = true;
#endif

    // Pick Whisper model
    info.suggestedModel = chooseWhisperModel(info);

    // Choose audio output
    selectOutputDevice(info);

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
        if (info.gpuVRAM_MB > 0)
            std::cout << "VRAM: " << info.gpuVRAM_MB << " MB\n";
        if (!info.gpuDriver.empty())
            std::cout << "Driver: " << info.gpuDriver << "\n";
        if (info.hasCUDA)  std::cout << "CUDA supported.\n";
        if (info.hasMetal) std::cout << "Metal supported.\n";
        if (info.hasROCm)  std::cout << "ROCm supported.\n";
    } else {
        std::cout << "No GPU detected.\n";
    }

    std::cout << "Voice backends:\n";
    std::cout << "  Windows SAPI: " << (info.hasSAPI ? "Yes" : "No") << "\n";
    std::cout << "  macOS say:   " << (info.hasSay ? "Yes" : "No") << "\n";
    std::cout << "  Linux Piper: " << (info.hasPiper ? "Yes" : "No") << "\n";

    std::cout << "Default (input): " << sf::SoundBufferRecorder::getDefaultDevice() << "\n";
    std::cout << "Selected (output): " << info.outputDevice << "\n";

    std::cout << "Suggested Whisper model: " << info.suggestedModel << "\n";
    std::cout << "-------------------------------\n";
}

// =========================================================
// Whisper model chooser
// =========================================================
std::string chooseWhisperModel(const SystemInfo& info) {
    if (info.hasGPU && info.ramMB > 16000) return "large-v3";
    if (info.hasGPU && info.ramMB > 8000)  return "medium";
    if (info.ramMB > 4000)                 return "small";
    return "base.en";
}
