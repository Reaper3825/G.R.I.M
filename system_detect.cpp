#include "system_detect.hpp"
#include "logger.hpp"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <thread>
#include <iostream>
#include "core/audio_core.hpp"


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

#ifdef WHISPER_USE_CUDA
    #include <cuda_runtime.h>
#endif

#ifdef _WIN32
#include <Wlanapi.h>
#pragma comment(lib, "wlanapi.lib")

LocationInfo g_location;

// =========================================================
// GeoIP location fetch (SAFE CPR version)
// =========================================================
static bool fetchLocationByIP(LocationInfo& loc) {
    try {
        LOG_DEBUG("SystemDetect", "Starting IP geolocation request...");

        // --- Initialize CPR Session explicitly ---
        cpr::Session session;
        session.SetOption(cpr::Redirect{true});
        session.SetUrl(cpr::Url{"http://ip-api.com/json/"}); 


        session.SetTimeout(cpr::Timeout{5000});   // 5 s timeout

        // --- Perform GET safely ---
        cpr::Response r = session.Get();
        

        // --- Check for network or connection error ---
        if (r.error) {
            LOG_ERROR("SystemDetect",
                std::string("CPR error: ") +
                r.error.message + " (code " + std::to_string((int)r.error.code) + ")");
            return false;
        }

        // --- Validate HTTP status ---
        if (r.status_code != 200) {
            LOG_ERROR("SystemDetect",
                "Location fetch failed: HTTP " + std::to_string(r.status_code));
            return false;
        }

        // --- Parse response safely ---
        nlohmann::json j = nlohmann::json::parse(r.text, nullptr, false);
        if (j.is_discarded()) {
            LOG_ERROR("SystemDetect", "Invalid JSON in IP API response.");
            return false;
        }

        if (!j.contains("status") || j["status"] != "success") {
            std::string msg = j.contains("message") ? j["message"].get<std::string>() : "unknown";
            LOG_ERROR("SystemDetect", "GeoIP lookup failed: " + msg);
            return false;
        }

        // --- Populate LocationInfo ---
        loc.ip      = j.value("query", "");
        loc.country = j.value("country", "");
        loc.region  = j.value("regionName", "");
        loc.city    = j.value("city", "");
        loc.postal  = j.value("zip", "");
        loc.lat     = j.value("lat", 0.0);
        loc.lon     = j.value("lon", 0.0);
        loc.isp     = j.value("isp", "");

        LOG_DEBUG("SystemDetect", "GeoIP success: " + loc.fullAddress());
        return true;
    }
    catch (const nlohmann::json::exception& e) {
        LOG_ERROR("SystemDetect", std::string("JSON parse exception: ") + e.what());
    }
    catch (const std::exception& e) {
        LOG_ERROR("SystemDetect", std::string("CPR/Network exception: ") + e.what());
    }
    catch (...) {
        LOG_ERROR("SystemDetect", "Unknown fatal exception in fetchLocationByIP.");
    }

    return false;
}

// =========================================================
// Wi-Fi connection detection
// =========================================================
bool g_wifiConnected = false;
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
    for (int i = 0; i < (int)pIfList->dwNumberOfItems; i++) {
        WLAN_INTERFACE_INFO info = pIfList->InterfaceInfo[i];
        if (info.isState == wlan_interface_state_connected) {
            connected = true;
            break;
        }
    }

    if (pIfList != nullptr) WlanFreeMemory(pIfList);
    WlanCloseHandle(hClient, NULL);
    return connected;
}
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

static bool detectWindowsGPU(SystemInfo& info);
static void detectMonitors(SystemInfo& info);
#endif

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
                                        if (vtDriver.vt == VT_BSTR)
                                            info.gpuDriver = wideToUtf8(vtDriver.bstrVal);
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

#ifdef _WIN32
BOOL CALLBACK MonitorEnumProc(HMONITOR hMon, HDC, LPRECT lprcMonitor, LPARAM dwData) {
    auto* info = reinterpret_cast<SystemInfo*>(dwData);

    MONITORINFOEX mi;
    mi.cbSize = sizeof(mi);
    if (GetMonitorInfo(hMon, &mi)) {
        MonitorInfo m;
        m.x = mi.rcMonitor.left;
        m.y = mi.rcMonitor.top;
        m.width  = mi.rcMonitor.right  - mi.rcMonitor.left;
        m.height = mi.rcMonitor.bottom - mi.rcMonitor.top;
        m.isPrimary = (mi.dwFlags & MONITORINFOF_PRIMARY);

        info->monitors.push_back(m);
        info->monitorCount++;
        info->hasMonitor = true;

        info->totalScreenWidth  = std::max<int>(info->totalScreenWidth,  mi.rcMonitor.right);
        info->totalScreenHeight = std::max<int>(info->totalScreenHeight, mi.rcMonitor.bottom);
    }

    return TRUE;
}

static void detectMonitors(SystemInfo& info) {
    info.monitors.clear();
    info.monitorCount = 0;
    info.totalScreenWidth = 0;
    info.totalScreenHeight = 0;
    info.hasMonitor = false;

    EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, reinterpret_cast<LPARAM>(&info));
}
#endif

static void selectOutputDevice(SystemInfo& info) {
    info.outputDevice = Audio::getDefaultOutput();
    LOG_PHASE("Output device defaulted", true);
    LOG_DEBUG("SystemDetect", "Using default output device: " + info.outputDevice);
}



// =========================================================
// Main detection entry
// =========================================================
SystemInfo detectSystem() {
    SystemInfo info;

#ifdef _WIN32
    info.osName = "Windows";
    info.hasSAPI = true;
    g_wifiConnected = detectWifiConnected();
#elif __APPLE__
    info.osName = "macOS";
    info.hasSay = true;
#elif __linux__
    info.osName = "Linux";
    info.hasPiper = ensurePiperInstalled();
#endif

    LOG_DEBUG("SystemDetect", "Wi-Fi connected: " + std::string(g_wifiConnected ? "Yes" : "No"));

#if defined(__x86_64__) || defined(_M_X64)
    info.arch = "x86_64";
#elif defined(__aarch64__)
    info.arch = "ARM64";
#elif defined(__arm__)
    info.arch = "ARM";
#else
    info.arch = "Unknown";
#endif

    info.cpuCores = std::thread::hardware_concurrency();

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
    if (sysinfo(&sys) == 0)
        info.ramMB = sys.totalram / (1024 * 1024);
#endif

#ifdef WHISPER_USE_CUDA
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

#ifdef _WIN32
    if (!info.hasGPU) detectWindowsGPU(info);
    detectMonitors(info);
#endif

#ifdef __APPLE__
    info.hasGPU = true;
    info.hasMetal = true;
#endif

    info.suggestedModel = chooseWhisperModel(info);

    if (g_wifiConnected) {
        if (!fetchLocationByIP(g_location))
            LOG_DEBUG("SystemDetect", "IP geolocation failed or timed out.");
        else
            LOG_DEBUG("SystemDetect", "Location: " + g_location.fullAddress());
    } else {
        LOG_DEBUG("SystemDetect", "Skipping location fetch — Wi-Fi not connected.");
    }

    selectOutputDevice(info);
    return info;
}
// =========================================================
// Logging
// =========================================================
void logSystemInfo(const SystemInfo& info) {
    LOG_PHASE("---- GRIM System Detection ----", true);

    LOG_DEBUG("SystemDetect", "OS: " + info.osName + " (" + info.arch + ")");
    LOG_DEBUG("SystemDetect", "CPU cores: " + std::to_string(info.cpuCores));
    LOG_DEBUG("SystemDetect", "RAM: " + std::to_string(info.ramMB) + " MB");

    if (info.hasGPU) {
        LOG_DEBUG("SystemDetect", "GPU detected: " + info.gpuName +
            " (" + std::to_string(info.gpuCount) + " device(s))");

        if (info.gpuVRAM_MB > 0)
            LOG_DEBUG("SystemDetect", "VRAM: " + std::to_string(info.gpuVRAM_MB) + " MB");

        if (!info.gpuDriver.empty())
            LOG_DEBUG("SystemDetect", "Driver: " + info.gpuDriver);

        if (info.hasCUDA)  LOG_DEBUG("SystemDetect", "CUDA supported.");
        if (info.hasMetal) LOG_DEBUG("SystemDetect", "Metal supported.");
        if (info.hasROCm)  LOG_DEBUG("SystemDetect", "ROCm supported.");
    } else {
        LOG_DEBUG("SystemDetect", "No GPU detected.");
    }

    LOG_DEBUG("SystemDetect", "Voice backends:");
    LOG_DEBUG("SystemDetect", "  Windows SAPI: " + std::string(info.hasSAPI ? "Yes" : "No"));
    LOG_DEBUG("SystemDetect", "  macOS say:   " + std::string(info.hasSay ? "Yes" : "No"));
    LOG_DEBUG("SystemDetect", "  Linux Piper: " + std::string(info.hasPiper ? "Yes" : "No"));

    LOG_DEBUG("SystemDetect", "Default input: " + Audio::getDefaultInput());
    LOG_DEBUG("SystemDetect", "Default output: " + info.outputDevice);


    if (info.hasMonitor) {
        LOG_DEBUG("SystemDetect", "Monitors detected: " + std::to_string(info.monitorCount));
        for (size_t i = 0; i < info.monitors.size(); ++i) {
            const auto& m = info.monitors[i];
            LOG_DEBUG("SystemDetect",
                "  Monitor " + std::to_string(i) +
                " [" + std::to_string(m.width) + "x" + std::to_string(m.height) +
                " @(" + std::to_string(m.x) + "," + std::to_string(m.y) + ")]" +
                (m.isPrimary ? " [PRIMARY]" : "")
            );
        }
        LOG_DEBUG("SystemDetect", "Virtual desktop bounds: " +
            std::to_string(info.totalScreenWidth) + "x" +
            std::to_string(info.totalScreenHeight));
    } else {
        LOG_DEBUG("SystemDetect", "No monitors detected.");
    }

    LOG_DEBUG("SystemDetect", "Suggested Whisper model: " + info.suggestedModel);
    LOG_PHASE("-------------------------------", true);
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
