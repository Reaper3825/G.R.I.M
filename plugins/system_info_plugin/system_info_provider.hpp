#pragma once
#include "core/plugin_api.hpp"
#include "MMO/Core/HardwareInventoryJson.hpp"
#include <functional>
#include <mutex>

namespace GRIM::SystemInfo {
// Implemented inside the plugin; no dependency on the host startup detector or audio service.
MMO::HardwareInventory detectNativeHardware();
struct UnknownDetectionFailure {};
class Provider {
public:
    using Detector = std::function<MMO::HardwareInventory()>;
    explicit Provider(const GrimPluginAPI* api, Detector detector = detectNativeHardware)
        : api_(api), detector_(std::move(detector)) {}
    // Returns this attempt, publishes copied resources, retains last-good data on probe failure.
    nlohmann::json refresh();
    MMO::HardwareInventory detectHardware();
private:
    const GrimPluginAPI* api_;
    Detector detector_;
    std::mutex mutex_;
    nlohmann::json lastHardware_ = nullptr;
};
}
