#pragma once
#include "core/plugin_api.hpp"
#include "MMO/Core/HardwareInventoryJson.hpp"
#include "commands/command_result.hpp"
#include <stdexcept>

namespace hostHardware {
inline const GrimPluginAPI* host = nullptr;
struct UnknownDetectionFailure {};
inline GRIM::MMO::HardwareInventory detectHardware() {
    struct OwnedText {
        const char* text;
        ~OwnedText() { if (text) host->string_free(host, text); }
    } text{host->get_hardware_snapshot(host)};
    if (!text.text) throw std::runtime_error("Host hardware snapshot unavailable");
    const auto value = nlohmann::json::parse(text.text);
    if (value.at("format") != "grim-hardware-snapshot" || value.at("version") != 1)
        throw std::runtime_error("Unsupported host hardware snapshot");
    if (!value.at("error").is_null()) {
        if (value.at("error").at("kind") == "exception")
            throw std::runtime_error(value.at("error").at("message").get<std::string>());
        throw UnknownDetectionFailure{};
    }
    return GRIM::MMO::hardwareInventoryFromJson(value.at("hardware"));
}
}
CommandResult cmdSystemInfo(const std::string&);
