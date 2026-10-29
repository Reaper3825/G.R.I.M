#include "plugins/runtime_service_client.hpp"

namespace {
template<GrimPerceptionQuery Query>
GrimResult call(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    // All three former handlers ignored their arguments. The host performs the
    // same availability check and calls its existing observation service once.
    const auto value = grim::runtime::response(grim::runtime::host->query_perception(grim::runtime::host, Query));
    if (!value.at("available").get<bool>()) {
        return grim::runtime::deliver(writer, {false, "[Perception] System not initialized",
            "ERR_PERCEPTION_NOT_INITIALIZED", "error", "Perception system not available", Colors::Red});
    }
    const auto text = value.at("text").get<std::string>();
    // Preserve literal output and success status, including empty/error text.
    return grim::runtime::deliver(writer, {true, text, "ERR_NONE", "routine", text, Colors::Cyan});
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Perception Plugin", "1.0.0", "GRIM Project",
        "Screen analysis, OCR, and object observations through existing host perception services", GRIM_PERM_SYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->query_perception) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::runtime::host = api;
    GrimToolRegistration tools[3];
    tools[0].handler = call<GRIM_PERCEPTION_ANALYZE_SCREEN>;
    tools[1].handler = call<GRIM_PERCEPTION_READ_TEXT>;
    tools[2].handler = call<GRIM_PERCEPTION_DETECT_OBJECTS>;
    const char* ids[] = {"perception_what_see", "perception_read_text", "perception_detect_objects"};
    const char* descriptions[] = {"Describe what's visible on screen", "Read text from screen using OCR", "Detect and locate objects on screen"};
    const char* contexts[] = {"Host perception subsystem initialized"};
    for (size_t i = 0; i < 3; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = d.usage = ids[i];
        d.version = "1.0.0"; d.description = descriptions[i]; d.category = "information";
        d.argument_format = "raw_text"; d.required_permissions = GRIM_PERM_SYSTEM; d.is_informational = 1;
        d.examples = {&ids[i], 1}; d.preconditions = {contexts, 1};
    }
    return api->register_tools(api, tools, 3);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { grim::runtime::host = nullptr; }
