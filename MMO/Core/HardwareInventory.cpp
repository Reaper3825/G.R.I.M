// Multi-Model Orchestration (MMO) - Hardware Inventory Implementation
// Replaces system_detect.cpp detection logic.
//======================================================//
#include "HardwareInventory.hpp"
#include "logger.hpp"
#include "core/audio_core.hpp"

#include <thread>
#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
    #include <windows.h>  // GetPhysicallyInstalledSystemMemory, etc.
    #include <dxgi.h>
    #include <Wbemidl.h>
    #include <comdef.h>
    #include <Wlanapi.h>
    #pragma comment(lib, "ole32.lib")
    #pragma comment(lib, "wbemuuid.lib")
    #pragma comment(lib, "dxgi.lib")
    #pragma comment(lib, "wlanapi.lib")
#elif __APPLE__
    #include <sys/types.h>
    #include <sys/sysctl.h>
#elif __linux__
    #include <sys/sysinfo.h>
    #include <unistd.h>
#endif

#ifdef WHISPER_USE_CUDA
    #include <cuda_runtime.h>
#endif

namespace GRIM::MMO {

// =========================================================
// Platform helpers (Windows)
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

static bool detectWifiConnected() {
    HANDLE hClient = nullptr;
    DWORD dwMaxClient = 2;
    DWORD dwCurVersion = 0;
    DWORD dwResult = WlanOpenHandle(dwMaxClient, NULL, &dwCurVersion, &hClient);
    if (dwResult != ERROR_SUCCESS) return false;

    PWLAN_INTERFACE_INFO_LIST pIfList = nullptr;
    dwResult = WlanEnumInterfaces(hClient, NULL, &pIfList);
    if (dwResult != ERROR_SUCCESS) {
        WlanCloseHandle(hClient, NULL);
        return false;
    }

    bool connected = false;
    for (DWORD i = 0; i < pIfList->dwNumberOfItems; i++) {
        if (pIfList->InterfaceInfo[i].isState == wlan_interface_state_connected) {
            connected = true;
            break;
        }
    }

    if (pIfList) WlanFreeMemory(pIfList);
    WlanCloseHandle(hClient, NULL);
    return connected;
}

static void detectWindowsGPUs(HardwareInventory& inv) {
    IDXGIFactory* pFactory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory)))
        return;

    IDXGIAdapter* pAdapter = nullptr;
    int index = 0;

    while (pFactory->EnumAdapters(index, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
        DXGI_ADAPTER_DESC desc;
        if (SUCCEEDED(pAdapter->GetDesc(&desc))) {
            std::wstring ws(desc.Description);
            std::string name(ws.begin(), ws.end());

            if (name.find("NVIDIA") != std::string::npos ||
                name.find("AMD")    != std::string::npos ||
                name.find("Intel")  != std::string::npos) {

                GPUDevice gpu;
                gpu.device_index = index;
                gpu.name         = name;
                gpu.vram_mb      = static_cast<long>(desc.DedicatedVideoMemory / (1024 * 1024));
                inv.gpus.push_back(gpu);
            }
        }
        pAdapter->Release();
        index++;
    }
    pFactory->Release();

    inv.gpu_count = static_cast<int>(inv.gpus.size());
    if (inv.gpu_count == 0) return;

    // Query driver version via WMI for NVIDIA GPUs
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
                if (SUCCEEDED(pSvc->ExecQuery(
                        bstr_t("WQL"),
                        bstr_t("SELECT Name, DriverVersion FROM Win32_VideoController"),
                        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                        NULL, &pEnumerator))) {
                    IWbemClassObject* pclsObj = nullptr;
                    ULONG uReturn = 0;
                    while (pEnumerator &&
                           pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn) == S_OK) {
                        VARIANT vtName;
                        if (SUCCEEDED(pclsObj->Get(L"Name", 0, &vtName, 0, 0))) {
                            if (vtName.vt == VT_BSTR) {
                                std::string wmiName = wideToUtf8(vtName.bstrVal);
                                // Match WMI name to our DXGI-detected GPUs
                                for (auto& gpu : inv.gpus) {
                                    if (wmiName.find(gpu.name.substr(0, 10)) != std::string::npos ||
                                        gpu.name.find(wmiName.substr(0, 10)) != std::string::npos) {
                                        VARIANT vtDriver;
                                        if (SUCCEEDED(pclsObj->Get(L"DriverVersion", 0, &vtDriver, 0, 0))) {
                                            if (vtDriver.vt == VT_BSTR)
                                                gpu.driver_version = wideToUtf8(vtDriver.bstrVal);
                                            VariantClear(&vtDriver);
                                        }
                                    }
                                }
                            }
                            VariantClear(&vtName);
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
}

struct MonitorEnumContext {
    HardwareInventory* inv;
};

static BOOL CALLBACK MonitorEnumProc(HMONITOR hMon, HDC, LPRECT, LPARAM dwData) {
    auto* ctx = reinterpret_cast<MonitorEnumContext*>(dwData);
    auto& inv = *ctx->inv;

    MONITORINFOEX mi;
    mi.cbSize = sizeof(mi);
    if (GetMonitorInfo(hMon, &mi)) {
        MonitorInfo m;
        m.x          = mi.rcMonitor.left;
        m.y          = mi.rcMonitor.top;
        m.width      = mi.rcMonitor.right  - mi.rcMonitor.left;
        m.height     = mi.rcMonitor.bottom - mi.rcMonitor.top;
        m.is_primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;

        inv.monitors.push_back(m);
        inv.monitor_count++;

        inv.virtual_desktop_w = std::max<int>(inv.virtual_desktop_w, mi.rcMonitor.right);
        inv.virtual_desktop_h = std::max<int>(inv.virtual_desktop_h, mi.rcMonitor.bottom);
    }
    return TRUE;
}

static void detectMonitors(HardwareInventory& inv) {
    inv.monitors.clear();
    inv.monitor_count     = 0;
    inv.virtual_desktop_w = 0;
    inv.virtual_desktop_h = 0;

    MonitorEnumContext ctx{&inv};
    EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, reinterpret_cast<LPARAM>(&ctx));

    inv.virtual_origin_x  = GetSystemMetrics(SM_XVIRTUALSCREEN);
    inv.virtual_origin_y  = GetSystemMetrics(SM_YVIRTUALSCREEN);
    inv.virtual_desktop_w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    inv.virtual_desktop_h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
}

#endif // _WIN32

#ifdef __linux__
static bool commandExists(const char* cmd) {
    std::string check = "which " + std::string(cmd) + " > /dev/null 2>&1";
    return (system(check.c_str()) == 0);
}
#endif

// =========================================================
// Whisper model chooser (same logic as old chooseWhisperModel)
// =========================================================
static std::string chooseWhisperModel(const HardwareInventory& inv) {
    bool hasGPU = inv.gpu_count > 0;
    if (hasGPU && inv.ram_total_mb > 16000) return "large-v3";
    if (hasGPU && inv.ram_total_mb > 8000)  return "medium";
    if (inv.ram_total_mb > 4000)            return "small";
    return "base.en";
}

// =========================================================
// Main detection entry — replaces detectSystem()
// =========================================================
HardwareInventory detectHardware() {
    HardwareInventory inv;
    inv.capture_time = std::chrono::steady_clock::now();

    // --- OS ---
#ifdef _WIN32
    inv.os_name  = "Windows";
    inv.has_sapi = true;
    inv.wifi_connected = detectWifiConnected();
#elif __APPLE__
    inv.os_name = "macOS";
    inv.has_say = true;
#elif __linux__
    inv.os_name  = "Linux";
    inv.has_piper = commandExists("piper");
#endif

    // --- Arch ---
#if defined(__x86_64__) || defined(_M_X64)
    inv.arch = "x86_64";
#elif defined(__aarch64__)
    inv.arch = "ARM64";
#elif defined(__arm__)
    inv.arch = "ARM";
#else
    inv.arch = "Unknown";
#endif

    // --- CPU ---
    inv.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());

    // --- RAM ---
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    inv.ram_total_mb = static_cast<long>(status.ullTotalPhys / (1024 * 1024));
#elif __APPLE__
    int64_t mem;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, NULL, 0);
    inv.ram_total_mb = static_cast<long>(mem / (1024 * 1024));
#elif __linux__
    struct sysinfo sys;
    if (sysinfo(&sys) == 0)
        inv.ram_total_mb = static_cast<long>(sys.totalram / (1024 * 1024));
#endif

    // --- GPU (CUDA runtime path) ---
#ifdef WHISPER_USE_CUDA
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            GPUDevice gpu;
            gpu.device_index = i;
            gpu.name         = prop.name;
            gpu.vram_mb      = static_cast<long>(prop.totalGlobalMem / (1024 * 1024));
            gpu.has_cuda     = true;
            inv.gpus.push_back(gpu);
        }
        inv.gpu_count = deviceCount;
    }
#endif

    // --- GPU (DXGI fallback on Windows when CUDA didn't populate) ---
#ifdef _WIN32
    if (inv.gpu_count == 0)
        detectWindowsGPUs(inv);

    // Mark CUDA on any NVIDIA GPU if CUDA runtime is present
#ifdef WHISPER_USE_CUDA
    for (auto& gpu : inv.gpus)
        if (gpu.name.find("NVIDIA") != std::string::npos)
            gpu.has_cuda = true;
#endif

    detectMonitors(inv);
#endif

#ifdef __APPLE__
    if (inv.gpu_count == 0) {
        GPUDevice gpu;
        gpu.device_index = 0;
        gpu.name         = "Apple Metal GPU";
        gpu.has_metal    = true;
        inv.gpus.push_back(gpu);
        inv.gpu_count = 1;
    }
#endif

    // --- Audio ---
    inv.audio_output_device = Audio::getDefaultOutput();
    inv.audio_input_device  = Audio::getDefaultInput();

    // --- Whisper suggestion ---
    inv.suggested_whisper_model = chooseWhisperModel(inv);

    return inv;
}

// =========================================================
// Logging
// =========================================================
void logHardwareInventory(const HardwareInventory& inv) {
    LOG_PHASE("---- Hardware Inventory ----", true);

    LOG_DEBUG("HardwareInventory", "OS: " + inv.os_name + " (" + inv.arch + ")");
    LOG_DEBUG("HardwareInventory", "CPU cores: " + std::to_string(inv.cpu_cores));
    LOG_DEBUG("HardwareInventory", "RAM: " + std::to_string(inv.ram_total_mb) + " MB");

    if (inv.gpu_count > 0) {
        LOG_DEBUG("HardwareInventory", "GPUs detected: " + std::to_string(inv.gpu_count));
        for (size_t i = 0; i < inv.gpus.size(); ++i) {
            const auto& gpu = inv.gpus[i];
            std::string line = "  GPU " + std::to_string(i) + ": " + gpu.name;
            if (gpu.vram_mb > 0)
                line += " (" + std::to_string(gpu.vram_mb) + " MB)";
            if (!gpu.driver_version.empty())
                line += " driver=" + gpu.driver_version;
            if (gpu.has_cuda)  line += " [CUDA]";
            if (gpu.has_metal) line += " [Metal]";
            if (gpu.has_rocm)  line += " [ROCm]";
            LOG_DEBUG("HardwareInventory", line);
        }
    } else {
        LOG_DEBUG("HardwareInventory", "No GPU detected.");
    }

    LOG_DEBUG("HardwareInventory", "Voice backends: SAPI=" +
        std::string(inv.has_sapi ? "Y" : "N") + " say=" +
        std::string(inv.has_say ? "Y" : "N") + " Piper=" +
        std::string(inv.has_piper ? "Y" : "N"));

    LOG_DEBUG("HardwareInventory", "Audio output: " + inv.audio_output_device);
    LOG_DEBUG("HardwareInventory", "Audio input: " + inv.audio_input_device);

    if (inv.monitor_count > 0) {
        LOG_DEBUG("HardwareInventory", "Monitors: " + std::to_string(inv.monitor_count));
        for (size_t i = 0; i < inv.monitors.size(); ++i) {
            const auto& m = inv.monitors[i];
            LOG_DEBUG("HardwareInventory",
                "  Monitor " + std::to_string(i) +
                " [" + std::to_string(m.width) + "x" + std::to_string(m.height) +
                " @(" + std::to_string(m.x) + "," + std::to_string(m.y) + ")]" +
                (m.is_primary ? " [PRIMARY]" : ""));
        }
        LOG_DEBUG("HardwareInventory", "Virtual desktop: " +
            std::to_string(inv.virtual_desktop_w) + "x" +
            std::to_string(inv.virtual_desktop_h));
    } else {
        LOG_DEBUG("HardwareInventory", "No monitors detected.");
    }

    LOG_DEBUG("HardwareInventory", "Wi-Fi: " + std::string(inv.wifi_connected ? "connected" : "disconnected"));
    LOG_DEBUG("HardwareInventory", "Suggested Whisper model: " + inv.suggested_whisper_model);
    LOG_PHASE("----------------------------", true);
}

} // namespace GRIM::MMO
