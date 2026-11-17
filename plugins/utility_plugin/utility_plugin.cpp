#include <string_view>
// Fixed-response utility commands.
#include "core/plugin_api.hpp"

namespace {
GrimResult clean(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    GrimToolResult result;
    result.success = 1;
    result.message = "[Utility] Console cleared.";
    result.error_code = "ERR_NONE";
    result.category = "routine";
    result.voice = "Console cleared";
    result.color_rgb = 0x64FF64; // Colors::Green
    return writer->complete(writer, &result);
}

GrimResult help(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    // Static legacy help excludes retired agent/admin commands.
    // Dynamic registry help is available separately through list_tools/tool_info.
    GrimToolResult result;
    result.success = 1;
    result.message =
        "[Help] Available commands:\n"
        "- remember <key> <value>\n"
        "- recall <key>\n"
        "- forget <key>\n"
        "- pwd\n"
        "- cd <dir>\n"
        "- ls\n"
        "- mkdir <dir>\n"
        "- rm <file>\n"
        "- sysinfo\n"
        "- clean\n"
        "- help\n"
        "- voice\n"
        "- voice_stream\n";
    result.error_code = "ERR_NONE";
    result.category = "summary";
    result.voice = "Help shown";
    result.color_rgb = 0x64FFFF; // Colors::Cyan
    return writer->complete(writer, &result);
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    return GRIM_OK;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "clean") return clean(request, writer, nullptr);
    if (id == "help") return help(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}
