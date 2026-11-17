#include <string_view>
#include "core/plugin_api.hpp"
#include "host_memory.hpp"
#include "plugins/plugin_logging.hpp"
namespace {
template<CommandResult (*Handler)(const std::string&)>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    return grim::runtime::deliver(writer, Handler(std::string(request->arguments, request->arguments_size)));
}
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->stage_manual_memory || !api->search_memory || !api->flush_memory_record_if_present)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::runtime::host = api;
    grim::plugin::setLoggingAPI(api);
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "remember") return call<cmdRemember>(request, writer, nullptr);
    if (id == "recall") return call<cmdRecall>(request, writer, nullptr);
    if (id == "forget") return call<cmdForget>(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI({}); grim::runtime::host = {};
}
