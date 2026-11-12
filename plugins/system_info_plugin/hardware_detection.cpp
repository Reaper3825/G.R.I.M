// Plugin-owned native hardware probes. No host services or startup cache access.
//======================================================//
#include "system_info_provider.hpp"

#include <thread>
#include <algorithm>
#include <cstdlib>

#ifdef _WIN32
    #include <windows.h>  // GetPhysicallyInstalledSystemMemory, etc.
    #include <dxgi.h>
    #include <Wbemidl.h>
    #include <comdef.h>
    #include <wrl/client.h>
    #include <memory>
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
    #include <dlfcn.h>
#endif


namespace GRIM::SystemInfo {
using MMO::HardwareInventory;
using MMO::GPUDevice;
using MMO::MonitorInfo;

// =========================================================
// Platform helpers (Windows)
// =========================================================
#ifdef _WIN32

static std::string wideToUtf8(const wchar_t* wstr) {
    if (!wstr) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
    if (size_needed <= 0) return "";
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &str[0], size_needed, NULL, NULL);
    str.pop_back();
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
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory> factory;
    if (FAILED(CreateDXGIFactory(IID_PPV_ARGS(&factory)))) throw std::runtime_error("Cannot enumerate GPUs");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter> adapter;
        const auto status = factory->EnumAdapters(index, &adapter);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(status)) throw std::runtime_error("GPU enumeration failed");
        DXGI_ADAPTER_DESC desc{};
        if (FAILED(adapter->GetDesc(&desc))) throw std::runtime_error("GPU metadata unavailable");
        const auto name = wideToUtf8(desc.Description);
        if (name.find("NVIDIA") != std::string::npos || name.find("AMD") != std::string::npos ||
            name.find("Intel") != std::string::npos) {
            GPUDevice gpu;
            gpu.device_index = static_cast<int>(index); gpu.name = name;
            gpu.vram_mb = static_cast<long>(desc.DedicatedVideoMemory / (1024 * 1024));
            inv.gpus.push_back(std::move(gpu));
        }
    }
    inv.gpu_count = static_cast<int>(inv.gpus.size());
    if (!inv.gpu_count) return;
    struct Apartment {
        HRESULT status = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        ~Apartment() { if (SUCCEEDED(status)) CoUninitialize(); }
    } apartment;
    if (FAILED(apartment.status) && apartment.status != RPC_E_CHANGED_MODE) return;
    ComPtr<IWbemLocator> locator;
    if (FAILED(CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&locator)))) return;
    ComPtr<IWbemServices> service;
    if (FAILED(locator->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), nullptr, nullptr, nullptr, 0, nullptr, nullptr, &service))) return;
    CoSetProxyBlanket(service.Get(), RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                     RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE);
    ComPtr<IEnumWbemClassObject> enumerator;
    if (FAILED(service->ExecQuery(_bstr_t(L"WQL"), _bstr_t(L"SELECT Name, DriverVersion FROM Win32_VideoController"),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &enumerator))) return;
    for (;;) {
        ComPtr<IWbemClassObject> object;
        ULONG count = 0;
        if (enumerator->Next(2000, 1, &object, &count) != S_OK || !count) break;
        struct Variant { VARIANT value; Variant() { VariantInit(&value); } ~Variant() { VariantClear(&value); } } name, driver;
        if (FAILED(object->Get(L"Name", 0, &name.value, nullptr, nullptr)) || name.value.vt != VT_BSTR) continue;
        if (FAILED(object->Get(L"DriverVersion", 0, &driver.value, nullptr, nullptr)) || driver.value.vt != VT_BSTR) continue;
        const auto wmiName = wideToUtf8(name.value.bstrVal);
        for (auto& gpu : inv.gpus)
            if (wmiName.find(gpu.name.substr(0, 10)) != std::string::npos ||
                gpu.name.find(wmiName.substr(0, 10)) != std::string::npos)
                gpu.driver_version = wideToUtf8(driver.value.bstrVal);
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
    if (!EnumDisplayMonitors(nullptr, nullptr, MonitorEnumProc, reinterpret_cast<LPARAM>(&ctx)))
        throw std::runtime_error("Monitor enumeration failed");

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

// Optional native driver discovery. The module is released before returning and no GPU context is created.
static void detectCudaGPUs(HardwareInventory& inv) {
#if defined(_WIN32) || defined(__linux__)
#ifdef _WIN32
    struct Driver {
        HMODULE handle = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        ~Driver() { if (handle) FreeLibrary(handle); }
        void* symbol(const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
    } driver;
#define GRIM_CUDA_CALL __stdcall
#else
    struct Driver {
        void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        ~Driver() { if (handle) dlclose(handle); }
        void* symbol(const char* name) { return dlsym(handle, name); }
    } driver;
#define GRIM_CUDA_CALL
#endif
    if (!driver.handle) return;
    const auto init = reinterpret_cast<int (GRIM_CUDA_CALL*)(unsigned)>(driver.symbol("cuInit"));
    const auto countDevices = reinterpret_cast<int (GRIM_CUDA_CALL*)(int*)>(driver.symbol("cuDeviceGetCount"));
    const auto getDevice = reinterpret_cast<int (GRIM_CUDA_CALL*)(int*, int)>(driver.symbol("cuDeviceGet"));
    const auto getName = reinterpret_cast<int (GRIM_CUDA_CALL*)(char*, int, int)>(driver.symbol("cuDeviceGetName"));
    const auto getMemory = reinterpret_cast<int (GRIM_CUDA_CALL*)(size_t*, int)>(driver.symbol("cuDeviceTotalMem_v2"));
#undef GRIM_CUDA_CALL
    if (!init || !countDevices || !getDevice || !getName || !getMemory || init(0) != 0) return;
    int count = 0;
    if (countDevices(&count) != 0 || count < 1 || count > 256) return;
    std::vector<GPUDevice> devices;
    for (int index = 0; index < count; ++index) {
        int device = 0; char name[256]{}; size_t memory = 0;
        if (getDevice(&device, index) != 0 || getName(name, sizeof(name), device) != 0 || getMemory(&memory, device) != 0)
            return; // Discard an incomplete optional probe and use native fallback.
        name[sizeof(name) - 1] = '\0';
        GPUDevice gpu;
        gpu.device_index = index; gpu.name = name; gpu.vram_mb = static_cast<long>(memory / (1024 * 1024));
        gpu.has_cuda = true; devices.push_back(std::move(gpu));
    }
    inv.gpus = std::move(devices); inv.gpu_count = count;
#else
    (void)inv;
#endif
}

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
HardwareInventory detectNativeHardware() {
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
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) throw std::runtime_error("Cannot query system memory");
    inv.ram_total_mb = static_cast<long>(status.ullTotalPhys / (1024 * 1024));
#elif __APPLE__
    int64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) != 0) throw std::runtime_error("Cannot query system memory");
    inv.ram_total_mb = static_cast<long>(mem / (1024 * 1024));
#elif __linux__
    struct sysinfo sys{};
    if (sysinfo(&sys) != 0) throw std::runtime_error("Cannot query system memory");
    inv.ram_total_mb = static_cast<long>((uint64_t(sys.totalram) * sys.mem_unit) / (1024 * 1024));
#endif

    // Probe the installed CUDA driver dynamically; no toolkit or host CUDA linkage.
    detectCudaGPUs(inv);
#ifdef _WIN32
    if (inv.gpu_count == 0) detectWindowsGPUs(inv);
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

    // Audio devices belong to the voice provider; this detector does not query host audio state.

    // --- Whisper suggestion ---
    inv.suggested_whisper_model = chooseWhisperModel(inv);

    return inv;
}


} // namespace GRIM::SystemInfo
