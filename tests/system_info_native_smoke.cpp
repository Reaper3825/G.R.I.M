#include "plugins/system_info_plugin/system_info_provider.hpp"
#include <iostream>
int main() try {
    const auto inventory = GRIM::SystemInfo::detectNativeHardware();
    if (inventory.os_name.empty() || inventory.cpu_cores <= 0 || inventory.ram_total_mb <= 0)
        throw std::runtime_error("Native detector returned incomplete system data");
    const auto encoded = GRIM::MMO::hardwareInventoryJson(inventory);
    if (GRIM::MMO::hardwareInventoryJson(GRIM::MMO::hardwareInventoryFromJson(encoded)) != encoded)
        throw std::runtime_error("Native inventory codec lost data");
    std::cout << "Plugin native detector: " << inventory.os_name << ", " << inventory.cpu_cores
              << " cores, " << inventory.gpu_count << " GPUs, " << inventory.monitor_count << " monitors; no host services linked\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
