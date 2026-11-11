#pragma once
#include "RegisteredTool.hpp"
#include "ResourceDescriptor.hpp"

namespace GRIM::MMO {
struct ToolRegistrySnapshot {
    uint64_t revision = 0;
    std::vector<ProviderDescriptor> providers;
    std::vector<RegisteredTool> tools;
    uint64_t resource_revision = 0;
    std::vector<ResourceDescriptor> resources;
};
}
