#pragma once
#include "ToolDescriptor.hpp"
#include <nlohmann/json.hpp>

namespace GRIM::MMO {
// Data only. Identity is (provider type, provider name, resource ID);
// provider_instance and runtime_id identify the currently live generation.
struct ResourceDescriptor {
    ToolProviderType provider_type = ToolProviderType::Plugin;
    std::string provider_name, provider_instance;
    std::string resource_id, kind, name;
    std::string parent_id; // Reserved for widget hierarchy; currently must be empty.
    uint64_t runtime_id = 0;
    nlohmann::json state = nlohmann::json::object();
    std::vector<std::string> operations;
    bool operator==(const ResourceDescriptor&) const = default;
};
inline void to_json(nlohmann::json& j, const ResourceDescriptor& r) {
    j = {{"provider_type", r.provider_type == ToolProviderType::Plugin ? "plugin" : "builtin"},
         {"provider_name", r.provider_name}, {"provider_instance", r.provider_instance},
         {"resource_id", r.resource_id}, {"kind", r.kind}, {"name", r.name},
         {"parent_id", r.parent_id}, {"runtime_id", r.runtime_id},
         {"state", r.state}, {"operations", r.operations}};
}
}
