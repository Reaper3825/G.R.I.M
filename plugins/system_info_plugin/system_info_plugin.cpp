#include <string_view>
#include "core/plugin_api.hpp"
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
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::plugin::setLoggingAPI(api);
    provider = std::make_unique<GRIM::SystemInfo::Provider>(api);
    try { provider->refresh(); } // Probe failure publishes explicit stale state; publication failure rejects init.
    catch (...) { return GRIM_ERROR_UNKNOWN; }
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "system_info") return call(request, writer, provider.get());
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    provider.reset();
    grim::plugin::setLoggingAPI({});
}
