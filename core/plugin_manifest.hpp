#pragma once

#include "plugin_api.hpp"
#include "../MMO/Core/Descriptors/ProviderDescriptor.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

namespace PluginAPI {

inline uint32_t readManifestPermissions(const nlohmann::json& permissions) {
    static const std::unordered_map<std::string, uint32_t> flags{
        {"filesystem", GRIM_PERM_FILESYSTEM}, {"process", GRIM_PERM_PROCESS},
        {"network", GRIM_PERM_NETWORK}, {"memory", GRIM_PERM_MEMORY},
        {"voice", GRIM_PERM_VOICE}, {"ui", GRIM_PERM_UI}, {"system", GRIM_PERM_SYSTEM},
        {"input", GRIM_PERM_INPUT}
    };
    if (!permissions.is_array()) throw std::invalid_argument("Expected a permission array");
    uint32_t mask = 0;
    for (const auto& permission : permissions)
        mask |= flags.at(permission.get<std::string>());
    return mask;
}

inline GRIM::MMO::ProviderDescriptor readManifestProvider(
    const nlohmann::json& manifest, const std::string& sourcePath) {
    const auto& schema = manifest.at("schema_version");
    if (!schema.is_number_integer() || schema != 2)
        throw std::invalid_argument("Unsupported plugin manifest schema version");
    const auto& abi = manifest.at("api_version");
    if (!abi.is_number_integer() || abi < (GRIM_API_VERSION_MAJOR << 16) || abi > GRIM_API_VERSION)
        throw std::invalid_argument("Plugin API version mismatch");

    GRIM::MMO::ProviderDescriptor provider;
    provider.name = manifest.at("provider").get<std::string>();
    provider.version = manifest.at("version").get<std::string>();
    for (const auto* field : {&provider.name, &provider.version}) {
        if (field->empty() || field->size() > 65536 || field->find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid plugin identity");
    }
    provider.path = sourcePath;
    provider.api_version = abi.get<uint32_t>();
    provider.required_permissions = readManifestPermissions(manifest.at("required_permissions"));
    return provider;
}

} // namespace PluginAPI
