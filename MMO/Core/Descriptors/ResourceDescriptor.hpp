#pragma once
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

namespace GRIM::MMO {
// Resource IDs are scoped to a provider generation. Runtime IDs distinguish live bindings.
struct ResourceDescriptor {
    std::string resource_id, kind, name;
    std::string parent_id; // Reserved for parent hierarchy; currently must be empty.
    uint64_t runtime_id = 0;
    nlohmann::json state = nlohmann::json::object();
    bool operator==(const ResourceDescriptor&) const = default;
};
inline void to_json(nlohmann::json& j, const ResourceDescriptor& r) {
    j = {{"resource_id", r.resource_id}, {"kind", r.kind}, {"name", r.name},
         {"parent_id", r.parent_id}, {"runtime_id", r.runtime_id},
         {"state", r.state}};
}
}
