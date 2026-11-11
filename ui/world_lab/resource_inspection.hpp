#pragma once
#include "MMO/Core/Descriptors/RegisteredResource.hpp"
#include <sstream>

namespace GRIM::WorldLab {
// Presentation only: inspection reads published state and never probes hardware.
inline std::string resourceInspectionFeedback(const MMO::RegisteredResource& resource) {
    const auto& state = resource.state;
    if (resource.kind != "system_info.snapshot" ||
        state.value("format", "") != "grim-system-info-resource" || state.value("version", 0) != 1)
        return nlohmann::json(resource).dump(2);

    std::ostringstream text;
    text << resource.name << '\n';
    const auto& data = state.at("data");
    if (data.is_null()) {
        text << resource.name << " information unavailable\nStatus: Unavailable";
    } else {
        const auto component = state.at("component").get<std::string>();
        const auto line = [&](const char* label, const char* key, const char* suffix = "") {
            text << label << ": ";
            const auto& value = data.at(key);
            if (value.is_string()) text << value.get<std::string>();
            else text << value;
            text << suffix << '\n';
        };
        if (component == "cpu") line("Logical cores", "cpu_cores");
        else if (component == "memory") line("Total RAM", "ram_total_mb", " MB");
        else if (component == "system") {
            line("OS", "os_name"); line("Architecture", "arch");
            text << "Wi-Fi: " << (data.at("wifi_connected").get<bool>() ? "Connected" : "Disconnected") << '\n';
            line("GPUs", "gpu_count"); line("Monitors", "monitor_count");
            const auto& desktop = data.at("desktop");
            text << "Virtual desktop: " << desktop.at("width") << " x " << desktop.at("height")
                 << " at (" << desktop.at("origin_x") << ", " << desktop.at("origin_y") << ")\n";
        } else if (component == "gpu") {
            line("Device index", "device_index"); line("Name", "name");
            line("Total VRAM", "vram_mb", " MB");
            text << "Driver: " << (data.at("driver_version").get<std::string>().empty() ?
                "Unavailable" : data.at("driver_version").get<std::string>()) << '\n';
            for (const auto& capability : {std::pair{"CUDA", "has_cuda"},
                                          std::pair{"Metal", "has_metal"}, std::pair{"ROCm", "has_rocm"}})
                text << capability.first << ": " << (data.at(capability.second).get<bool>() ? "Supported" : "Not reported") << '\n';
        } else if (component == "monitor") {
            line("Monitor index", "monitor_index");
            text << "Position: (" << data.at("x") << ", " << data.at("y") << ")\n"
                 << "Dimensions: " << data.at("width") << " x " << data.at("height") << '\n'
                 << "Primary: " << (data.at("is_primary").get<bool>() ? "Yes" : "No") << '\n';
        } else return nlohmann::json(resource).dump(2);
        text << "Status: " << (state.at("stale").get<bool>() || !state.at("error").is_null() ?
            "Stale - latest capture failed." : "Current");
    }
    if (!state.at("error").is_null()) {
        const auto message = state.at("error").value("message", "");
        text << "\nCapture error: " << (message.empty() ? "Unknown capture failure" : message);
    }
    return text.str();
}
}
