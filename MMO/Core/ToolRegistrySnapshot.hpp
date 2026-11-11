#pragma once
#include "Descriptors/ProviderDescriptor.hpp"
#include "Descriptors/ResourceDescriptor.hpp"
#include "Descriptors/ToolDescriptor.hpp"
#include <cstddef>
#include <vector>

namespace GRIM::MMO {
struct ProviderSnapshot {
    ProviderDescriptor provider;
    std::vector<ToolDescriptor> tools;
    std::vector<ResourceDescriptor> resources;
    bool published = false; // A provider can also be discovered through standalone tool registrations.
};

struct ToolRegistrySnapshot {
    uint64_t revision = 0;
    uint64_t resource_revision = 0;
    std::vector<ProviderSnapshot> providers;

    std::size_t resourceCount() const {
        std::size_t count = 0;
        for (const auto& group : providers) count += group.resources.size();
        return count;
    }
};
}
