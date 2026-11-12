#include "system_info_provider.hpp"
#include "MMO/Core/Descriptors/ResourceDescriptor.hpp"
#include <stdexcept>
namespace GRIM::SystemInfo {
nlohmann::json Provider::refresh() {
    std::lock_guard lock(mutex_);
    nlohmann::json response{{"format", "grim-hardware-snapshot"}, {"version", 1},
        {"hardware", nullptr}, {"error", nullptr}};
    try {
        response["hardware"] = GRIM::MMO::hardwareInventoryJson(detector_());
    } catch (const std::exception& error) {
        response["error"] = {{"kind", "exception"}, {"message", error.what()}};
    } catch (...) {
        response["error"] = {{"kind", "unknown"}, {"message", ""}};
    }
    {
        const auto nextHardware = response["error"].is_null() ? response["hardware"] : lastHardware_;
        std::vector<GRIM::MMO::ResourceDescriptor> resources;
        const auto add = [&](const std::string& id, const std::string& component,
                             const std::string& name, nlohmann::json data) {
            GRIM::MMO::ResourceDescriptor resource;
            resource.resource_id = id;
            resource.kind = "system_info.snapshot";
            resource.name = name;
            // Identity includes resource_id; each object has one logical lifetime
            // within this provider generation, independent of changing readings.
            resource.runtime_id = 1;
            resource.state = {{"format", "grim-system-info-resource"}, {"version", 1},
                {"component", component}, {"data", std::move(data)},
                {"capture_time_ns", nextHardware.is_null() ? nlohmann::json(nullptr) :
                    nextHardware.at("capture_time_ns")},
                {"error", response["error"]}, {"stale", !response["error"].is_null()}};
            resources.push_back(std::move(resource));
        };
        const auto& h = nextHardware;
        if (h.is_null()) {
            // Always expose the fixed objects, even if the first probe failed.
            // Device objects require a successful enumeration.
            add("system", "system", "System", nullptr);
            add("cpu", "cpu", "CPU", nullptr);
            add("memory", "memory", "Memory", nullptr);
        } else {
            // Explicit field selection keeps audio and voice/model policy out of public objects.
            add("system", "system", "System", {{"os_name", h.at("os_name")},
                {"arch", h.at("arch")}, {"wifi_connected", h.at("wifi_connected")},
                {"gpu_count", h.at("gpu_count")}, {"monitor_count", h.at("monitor_count")},
                {"desktop", {{"origin_x", h.at("virtual_origin_x")},
                    {"origin_y", h.at("virtual_origin_y")},
                    {"width", h.at("virtual_desktop_w")}, {"height", h.at("virtual_desktop_h")}}}});
            add("cpu", "cpu", "CPU", {{"cpu_cores", h.at("cpu_cores")}});
            add("memory", "memory", "Memory", {{"ram_total_mb", h.at("ram_total_mb")}});
            for (const auto& gpu : h.at("gpus")) {
                const auto index = std::to_string(gpu.at("device_index").get<int>());
                add("gpu_" + index, "gpu", "GPU " + index + ": " + gpu.at("name").get<std::string>(),
                    {{"device_index", gpu.at("device_index")}, {"name", gpu.at("name")},
                     {"vram_mb", gpu.at("vram_mb")}, {"driver_version", gpu.at("driver_version")},
                     {"has_cuda", gpu.at("has_cuda")}, {"has_metal", gpu.at("has_metal")},
                     {"has_rocm", gpu.at("has_rocm")}});
            }
            size_t index = 0;
            for (const auto& monitor : h.at("monitors")) {
                const auto slot = std::to_string(index++);
                add("monitor_" + slot, "monitor", "Monitor " + slot,
                    {{"monitor_index", index - 1}, {"x", monitor.at("x")}, {"y", monitor.at("y")},
                     {"width", monitor.at("width")}, {"height", monitor.at("height")},
                     {"is_primary", monitor.at("is_primary")}});
            }
        }
        // Replace the whole capture atomically, removing devices no longer present.
        // Generation checks reject a retired provider's in-flight capture.
        const auto payload = nlohmann::json(resources).dump();
        if (api_->replace_resources(api_, "system_info.snapshot", payload.data(), payload.size()) != GRIM_OK)
            throw std::runtime_error("Hardware resource publication rejected");
        lastHardware_ = nextHardware;
    }
    return response;
}


MMO::HardwareInventory Provider::detectHardware() {
    const auto response = refresh();
    const auto& error = response.at("error");
    if (!error.is_null()) {
        if (error.at("kind") == "exception") throw std::runtime_error(error.at("message").get<std::string>());
        throw UnknownDetectionFailure{};
    }
    return MMO::hardwareInventoryFromJson(response.at("hardware"));
}
}
