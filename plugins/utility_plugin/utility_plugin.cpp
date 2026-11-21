#include <string_view>
// Fixed-response utility commands.
#include "core/plugin_api.hpp"
#include "core/plugin_tool_result.hpp"

namespace {
GrimToolReply clean(const GrimToolCall*, void*) {
    return grim::toolReply(true, "[Utility] Console cleared.",
                                    "ERR_NONE", "routine");
}

GrimToolReply help(const GrimToolCall*, void*) {
    // Static legacy help excludes retired agent/admin commands.
    // Dynamic registry help is available separately through list_tools/tool_info.
    const char* message =
        "[Help] Available commands:\n"
        "- remember <key> <value>\n"
        "- recall <key>\n"
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
    return grim::toolReply(true, message, "ERR_NONE", "summary");
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    return GRIM_OK;
}

extern "C" PLUGIN_EXPORT GrimToolReply grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request) {
    if (!tool_id) return {GRIM_ERROR_INVALID_PARAM};
    const std::string_view id(tool_id);
    if (id == "clean") return clean(request, nullptr);
    if (id == "help") return help(request, nullptr);
    return {GRIM_ERROR_NOT_FOUND};
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}
