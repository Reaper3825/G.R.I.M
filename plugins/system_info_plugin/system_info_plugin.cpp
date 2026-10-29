#include "host_hardware.hpp"
#include "plugins/plugin_logging.hpp"

namespace {
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    const auto value = cmdSystemInfo(std::string(request->arguments, request->arguments_size));
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
        "System information using fresh host hardware detection", GRIM_PERM_SYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_hardware_snapshot) return GRIM_ERROR_API_VERSION_MISMATCH;
    hostHardware::host = api;
    grim::plugin::setLoggingAPI(api);
    GrimToolRegistration tool;
    tool.handler = call;
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = "system_info";
    d.version = "1.0.0";
    d.description = "Display system hardware and software information";
    d.usage = "system_info";
    d.category = "information";
    d.argument_format = "raw_text";
    d.required_permissions = GRIM_PERM_SYSTEM;
    d.is_informational = 1;
    const char* examples[] = {"system_info"};
    d.examples = {examples, 1};
    return api->register_tools(api, &tool, 1);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI(nullptr);
    hostHardware::host = nullptr;
}
