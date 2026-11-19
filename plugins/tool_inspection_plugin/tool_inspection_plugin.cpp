#include <string_view>
#include "core/plugin_api.hpp"
#include "core/plugin_tool_result.hpp"
#include "snapshot_registry.hpp"
#include "plugins/plugin_logging.hpp"

namespace {
template<CommandResult (*Handler)(const std::string&)>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    const auto value = Handler(std::string(request->arguments, request->arguments_size));
    return grim::completeToolResult(writer, value);
}
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_tool_snapshot) return GRIM_ERROR_API_VERSION_MISMATCH;
    inspection::host = api;
    grim::plugin::setLoggingAPI(api);
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "list_tools") return call<cmdListTools>(request, writer, nullptr);
    if (id == "tool_info") return call<cmdToolInfo>(request, writer, nullptr);
    if (id == "tool_stats") return call<cmdToolStats>(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI({});
    inspection::host = {};
}
