#pragma once

#include "plugin_api.hpp"
#include "../MMO/Core/Descriptors/ToolDescriptorJson.hpp"

#include <new>
#include <string>
#include <vector>

namespace GRIM {

struct PluginToolRegistration {
    MMO::ToolDescriptor descriptor;
    GrimToolHandler handler = nullptr;
    void* user_data = nullptr;
};

inline GrimResult registerPluginTools(const GrimPluginAPI* api, const PluginToolRegistration* tools,
                                      size_t count) noexcept {
    if (!api || !api->register_tools || !tools || count == 0 || count > 256)
        return GRIM_ERROR_INVALID_PARAM;
    try {
        std::vector<std::string> metadata;
        std::vector<GrimToolRegistration> registrations;
        metadata.reserve(count);
        registrations.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& tool = tools[i];
            const auto& json = metadata.emplace_back(nlohmann::json(tool.descriptor).dump());
            registrations.push_back({sizeof(GrimToolRegistration), json.data(), json.size(),
                                     tool.handler, tool.user_data});
        }
        return api->register_tools(api, registrations.data(), registrations.size());
    } catch (const std::bad_alloc&) {
        return GRIM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return GRIM_ERROR_INVALID_PARAM;
    }
}

} // namespace GRIM
