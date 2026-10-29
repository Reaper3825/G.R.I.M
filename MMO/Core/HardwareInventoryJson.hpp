#pragma once
#include "HardwareInventory.hpp"
#include <nlohmann/json.hpp>

// Version-1 hardware snapshot codec. Clock ticks are encoded separately as
// monotonic nanoseconds; no chrono/STL object crosses the plugin ABI.
namespace GRIM::MMO {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MonitorInfo, x, y, width, height, is_primary)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GPUDevice, device_index, name, vram_mb, driver_version,
    has_cuda, has_metal, has_rocm)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(HardwareInventory, os_name, arch, cpu_cores, ram_total_mb,
    gpu_count, gpus, monitor_count, virtual_desktop_w, virtual_desktop_h, virtual_origin_x,
    virtual_origin_y, monitors, audio_output_device, audio_input_device, wifi_connected,
    has_sapi, has_say, has_piper, suggested_whisper_model)
inline nlohmann::json hardwareInventoryJson(const HardwareInventory& inventory) {
    nlohmann::json value = inventory;
    value["capture_time_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
        inventory.capture_time.time_since_epoch()).count();
    return value;
}
inline HardwareInventory hardwareInventoryFromJson(const nlohmann::json& value) {
    auto inventory = value.get<HardwareInventory>();
    inventory.capture_time = std::chrono::steady_clock::time_point{
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(value.at("capture_time_ns").get<int64_t>()))};
    return inventory;
}
}
