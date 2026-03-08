// Multi-Model Orchestration (MMO) - Resource Signal Implementation
// Platform-specific sampler loop for CPU, RAM, GPU utilisation.
//======================================================//
#include "ResourceSignal.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#elif __linux__
    #include <sys/sysinfo.h>
    #include <fstream>
    #include <sstream>
#elif __APPLE__
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#endif

// Optional: NVML for GPU utilisation on NVIDIA
#ifdef GRIM_HAS_NVML
    #include <nvml.h>
#endif

namespace GRIM::MMO {

// =========================================================
// Lifecycle
// =========================================================

ResourceSignal::~ResourceSignal() {
    stop();
}

void ResourceSignal::start(const ResourceSignalConfig& cfg, int gpu_count) {
    if (running_.load(std::memory_order_relaxed))
        return;

    config_    = cfg;
    gpu_count_ = gpu_count;

#ifdef GRIM_HAS_NVML
    nvmlInit();
#endif

    // Seed CPU baseline
#ifdef _WIN32
    FILETIME idle, kernel, user;
    if (GetSystemTimes(&idle, &kernel, &user)) {
        prev_idle_   = *(reinterpret_cast<uint64_t*>(&idle));
        prev_kernel_ = *(reinterpret_cast<uint64_t*>(&kernel));
        prev_user_   = *(reinterpret_cast<uint64_t*>(&user));
    }
#endif

    running_.store(true, std::memory_order_release);
    sampler_thread_ = std::thread(&ResourceSignal::samplerLoop, this);

    LOG_DEBUG("ResourceSignal", "Sampler started (poll=" +
              std::to_string(config_.poll_interval_ms) + "ms, gpus=" +
              std::to_string(gpu_count_) + ")");
}

void ResourceSignal::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel))
        return;

    if (sampler_thread_.joinable())
        sampler_thread_.join();

#ifdef GRIM_HAS_NVML
    nvmlShutdown();
#endif

    LOG_DEBUG("ResourceSignal", "Sampler stopped.");
}

ResourceSnapshot ResourceSignal::latest() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return snapshot_;
}

void ResourceSignal::forceUpdate() {
    force_update_.store(true, std::memory_order_release);
}

// =========================================================
// Sampler loop (runs on its own thread)
// =========================================================

int ResourceSignal::currentPollMs() const {
    // Read pressure from latest snapshot (under lock)
    PressureState p;
    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        p = snapshot_.pressure;
    }
    switch (p) {
        case PressureState::Critical:  return config_.critical_poll_interval_ms;
        case PressureState::Pressured: return config_.pressured_poll_interval_ms;
        default:                       return config_.poll_interval_ms;
    }
}

void ResourceSignal::samplerLoop() {
    while (running_.load(std::memory_order_acquire)) {
        ResourceSnapshot snap;
        snap.timestamp = std::chrono::steady_clock::now();

        sampleCPU(snap);
        sampleRAM(snap);
        sampleGPUs(snap);
        snap.pressure = derivePressure(snap);

        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            snapshot_ = std::move(snap);
        }

        // Sleep for the adaptive interval, waking early on forceUpdate
        int poll_ms = currentPollMs();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(poll_ms);
        while (std::chrono::steady_clock::now() < deadline &&
               running_.load(std::memory_order_relaxed)) {
            if (force_update_.exchange(false, std::memory_order_acq_rel))
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

// =========================================================
// CPU sampling
// =========================================================

void ResourceSignal::sampleCPU(ResourceSnapshot& snap) {
#ifdef _WIN32
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) {
        snap.cpu_utilization_pct = 0.0f;
        return;
    }

    uint64_t cur_idle   = *(reinterpret_cast<uint64_t*>(&idle));
    uint64_t cur_kernel = *(reinterpret_cast<uint64_t*>(&kernel));
    uint64_t cur_user   = *(reinterpret_cast<uint64_t*>(&user));

    uint64_t d_idle   = cur_idle   - prev_idle_;
    uint64_t d_kernel = cur_kernel - prev_kernel_;
    uint64_t d_user   = cur_user   - prev_user_;

    uint64_t d_total = d_kernel + d_user;
    if (d_total > 0) {
        // kernel time includes idle time on Windows
        snap.cpu_utilization_pct = 100.0f * static_cast<float>(d_total - d_idle)
                                          / static_cast<float>(d_total);
    }

    prev_idle_   = cur_idle;
    prev_kernel_ = cur_kernel;
    prev_user_   = cur_user;

#elif __linux__
    // Parse /proc/stat
    std::ifstream stat("/proc/stat");
    if (stat.is_open()) {
        std::string line;
        std::getline(stat, line);
        // cpu  user nice system idle iowait irq softirq steal
        unsigned long long u, n, s, idle_val, io, irq, sirq, steal;
        if (sscanf(line.c_str(), "cpu  %llu %llu %llu %llu %llu %llu %llu %llu",
                   &u, &n, &s, &idle_val, &io, &irq, &sirq, &steal) >= 4) {
            unsigned long long total = u + n + s + idle_val + io + irq + sirq + steal;
            unsigned long long busy  = total - idle_val - io;
            static unsigned long long prev_total = 0, prev_busy = 0;
            if (prev_total > 0 && total > prev_total) {
                snap.cpu_utilization_pct = 100.0f *
                    static_cast<float>(busy - prev_busy) /
                    static_cast<float>(total - prev_total);
            }
            prev_total = total;
            prev_busy  = busy;
        }
    }

#elif __APPLE__
    // macOS: host_statistics for CPU load
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long long total = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++)
            total += cpuinfo.cpu_ticks[i];
        unsigned long long idle_val = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
        static unsigned long long prev_total = 0, prev_idle = 0;
        if (prev_total > 0 && total > prev_total) {
            snap.cpu_utilization_pct = 100.0f *
                (1.0f - static_cast<float>(idle_val - prev_idle) /
                        static_cast<float>(total - prev_total));
        }
        prev_total = total;
        prev_idle  = idle_val;
    }
#endif
}

// =========================================================
// RAM sampling
// =========================================================

void ResourceSignal::sampleRAM(ResourceSnapshot& snap) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        long total = static_cast<long>(status.ullTotalPhys / (1024 * 1024));
        long avail = static_cast<long>(status.ullAvailPhys / (1024 * 1024));
        snap.ram_used_mb      = total - avail;
        snap.ram_available_mb = avail;
    }

#elif __linux__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        long total = static_cast<long>(si.totalram  * si.mem_unit / (1024 * 1024));
        long free  = static_cast<long>(si.freeram   * si.mem_unit / (1024 * 1024));
        snap.ram_used_mb      = total - free;
        snap.ram_available_mb = free;
    }

#elif __APPLE__
    int64_t total;
    size_t sz = sizeof(total);
    sysctlbyname("hw.memsize", &total, &sz, NULL, 0);
    long total_mb = static_cast<long>(total / (1024 * 1024));

    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    host_page_size(mach_port, &page_size);
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = sizeof(vm_stat) / sizeof(natural_t);
    if (host_statistics64(mach_port, HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        long free_mb = static_cast<long>(
            (vm_stat.free_count + vm_stat.inactive_count) * page_size / (1024 * 1024));
        snap.ram_used_mb      = total_mb - free_mb;
        snap.ram_available_mb = free_mb;
    }
#endif
}

// =========================================================
// GPU sampling (NVML when available, else zeroes)
// =========================================================

void ResourceSignal::sampleGPUs(ResourceSnapshot& snap) {
    snap.gpus.resize(gpu_count_);
    for (int i = 0; i < gpu_count_; ++i)
        snap.gpus[i].device_index = i;

#ifdef GRIM_HAS_NVML
    for (int i = 0; i < gpu_count_; ++i) {
        nvmlDevice_t dev;
        if (nvmlDeviceGetHandleByIndex(i, &dev) != NVML_SUCCESS)
            continue;

        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS)
            snap.gpus[i].utilization_pct = static_cast<float>(util.gpu);

        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
            snap.gpus[i].vram_used_mb = static_cast<long>(mem.used / (1024 * 1024));
            snap.gpus[i].vram_free_mb = static_cast<long>(mem.free / (1024 * 1024));
        }
    }
#endif
    // If NVML is not available, GPU live state stays zeroed.
    // Callers should check vram_free_mb == 0 as "unknown" rather than "full".
}

// =========================================================
// Pressure derivation
// =========================================================

PressureState ResourceSignal::derivePressure(const ResourceSnapshot& snap) const {
    PressureState worst = PressureState::Healthy;

    auto raise = [&worst](PressureState p) {
        if (static_cast<uint8_t>(p) > static_cast<uint8_t>(worst))
            worst = p;
    };

    // CPU
    if (snap.cpu_utilization_pct >= config_.cpu_critical_pct)
        raise(PressureState::Critical);
    else if (snap.cpu_utilization_pct >= config_.cpu_pressure_pct)
        raise(PressureState::Pressured);

    // RAM
    long ram_total = snap.ram_used_mb + snap.ram_available_mb;
    if (ram_total > 0) {
        float ram_pct = 100.0f * static_cast<float>(snap.ram_used_mb) /
                                 static_cast<float>(ram_total);
        if (ram_pct >= config_.ram_critical_pct)
            raise(PressureState::Critical);
        else if (ram_pct >= config_.ram_pressure_pct)
            raise(PressureState::Pressured);
    }

    // GPU VRAM (worst across all GPUs)
    for (const auto& gpu : snap.gpus) {
        long vram_total = gpu.vram_used_mb + gpu.vram_free_mb;
        if (vram_total <= 0) continue;  // unknown — don't count
        float vram_pct = 100.0f * static_cast<float>(gpu.vram_used_mb) /
                                   static_cast<float>(vram_total);
        if (vram_pct >= config_.vram_critical_pct)
            raise(PressureState::Critical);
        else if (vram_pct >= config_.vram_pressure_pct)
            raise(PressureState::Pressured);
    }

    return worst;
}

} // namespace GRIM::MMO
