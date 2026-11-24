#pragma once
#include "../Descriptors/ProviderDescriptor.hpp"
#include "../Descriptors/ResourceDescriptor.hpp"
#include "../Descriptors/ToolDescriptor.hpp"
#include <cstddef>
#include <vector>
#include <optional>

namespace GRIM::MMO {
struct ProviderSnapshot {
    ProviderDescriptor provider;
    std::vector<ToolDescriptor> tools;
    std::vector<ResourceDescriptor> resources;
};

// Present collections replace the provider's previous contents, including empty lists.
struct ProviderUpdate {
    std::string providerId;
    std::optional<std::vector<ToolDescriptor>> tools;
    std::optional<std::vector<ResourceDescriptor>> resources;
};

struct ProviderRegistrySnapshot {
    std::vector<ProviderSnapshot> providers;

    std::size_t resourceCount() const {
        std::size_t count = 0;
        for (const auto& group : providers) count += group.resources.size();
        return count;
    }
};
}
