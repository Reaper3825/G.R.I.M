// Multi-Model Orchestration (MMO) - Hardware Inventory
// Immutable startup snapshot of machine topology and capabilities.
// REPLACES the old SystemInfo struct. Built once during bootstrap
// by detectHardware(); never mutated after construction.
//======================================================//
#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace GRIM::MMO {

// =========================================================
// Per-monitor information
// =========================================================
struct MonitorInfo {
    int  x         = 0;   // top-left corner X (virtual desktop coords)
    int  y         = 0;   // top-left corner Y
    int  width     = 0;
    int  height    = 0;
    bool is_primary = false;
};

// =========================================================
// Per-GPU device record
// =========================================================
struct GPUDevice {
    int          device_index   = -1;
    std::string  name;
    long         vram_mb        = 0;
    std::string  driver_version;
    bool         has_cuda       = false;
    bool         has_metal      = false;
    bool         has_rocm       = false;
};

// =========================================================
// Immutable machine/topology snapshot (replaces SystemInfo)
// =========================================================
struct HardwareInventory {

    // --- Capture timestamp ---
    std::chrono::steady_clock::time_point capture_time{};

    // --- OS / Architecture ---
    std::string  os_name;     // "Windows", "macOS", "Linux"
    std::string  arch;        // "x86_64", "ARM64", etc.

    // --- CPU ---
    int          cpu_cores     = 0;

    // --- RAM ---
    long         ram_total_mb  = 0;

    // --- GPU inventory ---
    int          gpu_count     = 0;
    std::vector<GPUDevice> gpus;

    // --- Monitors ---
    int          monitor_count      = 0;
    int          virtual_desktop_w  = 0;
    int          virtual_desktop_h  = 0;
    int          virtual_origin_x   = 0;
    int          virtual_origin_y   = 0;
    std::vector<MonitorInfo> monitors;

    // --- Audio devices ---
    std::string  audio_output_device;
    std::string  audio_input_device;

    // --- Network ---
    bool         wifi_connected = false;

    // --- Voice backends ---
    bool         has_sapi  = false;
    bool         has_say   = false;
    bool         has_piper = false;

    // --- Whisper model suggestion ---
    std::string  suggested_whisper_model;

    // --- Convenience queries (derived from gpus vector) ---
    bool hasGPU()   const { return gpu_count > 0; }
    bool hasCUDA()  const { for (auto& g : gpus) if (g.has_cuda) return true; return false; }
    bool hasMetal() const { for (auto& g : gpus) if (g.has_metal) return true; return false; }
    bool hasROCm()  const { for (auto& g : gpus) if (g.has_rocm) return true; return false; }
};

// =========================================================
// Detect all hardware once during bootstrap.
// The result is immutable for the lifetime of the process.
// =========================================================
HardwareInventory detectHardware();

// Log the inventory to the standard GRIM logger.
void logHardwareInventory(const HardwareInventory& inv);

} // namespace GRIM::MMO
