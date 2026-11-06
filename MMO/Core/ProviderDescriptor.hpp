#pragma once
#include "ToolDescriptor.hpp"

namespace GRIM::MMO {
// Identity is (type, name); instance identifies one loaded generation.
// Providers exist independently of how many tools they expose.
struct ProviderDescriptor {
    ToolProviderType type = ToolProviderType::Plugin;
    std::string name;
    std::string instance;
    std::string path;
};
struct ToolRegistrySnapshot {
    uint64_t revision = 0;
    std::vector<ProviderDescriptor> providers;
    std::vector<ToolDescriptor> tools;
};
}
