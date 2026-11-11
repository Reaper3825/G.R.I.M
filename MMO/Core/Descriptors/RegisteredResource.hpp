#pragma once
#include "ProviderDescriptor.hpp"
#include "ResourceDescriptor.hpp"

namespace GRIM::MMO {
// Host inspection view copied from a single registry revision.
struct RegisteredResource : ResourceDescriptor {
    ProviderDescriptor provider;
    bool operator==(const RegisteredResource&) const = default;
};

inline void to_json(nlohmann::json& j, const RegisteredResource& resource) {
    to_json(j, static_cast<const ResourceDescriptor&>(resource));
    j["provider_type"] = "plugin";
    j["provider_name"] = resource.provider.name;
    j["provider_instance"] = resource.provider.instance;
}
}
