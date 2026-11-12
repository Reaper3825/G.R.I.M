// Linked into the fixture DLL only. Process environment is a test control channel;
// no host implementation or test exports are required by the production plugin.
#include "plugins/system_info_plugin/system_info_provider.hpp"
#include <windows.h>
#include <stdexcept>
namespace {
std::string setting(const char* key) {
    const auto size = GetEnvironmentVariableA(key, nullptr, 0);
    if (!size) throw std::runtime_error("Missing hardware fixture control");
    std::string value(size, '\0');
    GetEnvironmentVariableA(key, value.data(), size); value.pop_back(); return value;
}
}
namespace GRIM::SystemInfo {
MMO::HardwareInventory detectNativeHardware() {
    const auto count = std::stoi(setting("GRIM_SYSTEM_INFO_TEST_CALLS")) + 1;
    SetEnvironmentVariableA("GRIM_SYSTEM_INFO_TEST_CALLS", std::to_string(count).c_str());
    const auto mode = std::stoi(setting("GRIM_SYSTEM_INFO_TEST_MODE"));
    if (mode == 1) throw std::runtime_error("fixture detection failure");
    if (mode == 2) throw 42;
    return MMO::hardwareInventoryFromJson(nlohmann::json::parse(setting("GRIM_SYSTEM_INFO_TEST_INVENTORY")));
}
}
