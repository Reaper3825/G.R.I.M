#include "system_info_provider.hpp"
#include "system_info_command.hpp"
#include <memory>
#include "plugins/plugin_logging.hpp"

namespace {
std::unique_ptr<GRIM::SystemInfo::Provider> provider;
GrimResult call(const GrimToolCall*, GrimToolResultWriter* writer, void* context) {
    const auto value = formatSystemInfo(static_cast<GRIM::SystemInfo::Provider*>(context)->detectHardware());
    GrimToolResult result;
    result.success = value.success;
    result.message = value.message.c_str();
    result.error_code = value.errorCode.c_str();
    result.category = value.category.c_str();
    result.voice = value.voice.c_str();
    result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
    return writer->complete(writer, &result);
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "System Information Plugin", "1.0.0", "GRIM Project",
        "System information using plugin-owned hardware detection", GRIM_PERM_SYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::plugin::setLoggingAPI(api);
    GrimToolRegistration tool;
    provider = std::make_unique<GRIM::SystemInfo::Provider>(api);
    tool.handler = call;
    tool.user_data = provider.get();
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = "system_info";
    d.version = "1.0.0";
    d.description = "Display system hardware and software information";
    tool.short_description = "Display information about system hardware and software";
    d.usage = "system_info";
    d.category = "information";
    d.argument_format = "raw_text";
    d.required_permissions = GRIM_PERM_SYSTEM;
    d.is_informational = 1;
    const auto registered = api->register_tools(api, &tool, 1);
    if (registered != GRIM_OK) return registered;
    try { provider->refresh(); } // Probe failure publishes explicit stale state; publication failure rejects init.
    catch (...) { return GRIM_ERROR_UNKNOWN; }
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    provider.reset();
    grim::plugin::setLoggingAPI(nullptr);
}
