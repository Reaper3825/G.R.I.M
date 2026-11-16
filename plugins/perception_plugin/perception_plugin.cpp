#include <string_view>
#include "core/plugin_api.hpp"
#include "perception_runtime.hpp"
#include "plugins/plugin_logging.hpp"
#include <memory>
namespace {
std::unique_ptr<perceptionPlugin::Runtime> runtime;
const char* ids[] = {"perception.observe", "perception.digital.start", "perception.digital.stop",
    "perception.physical.open", "perception.physical.close", "perception.configure", "perception.calibrate"};
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, const char* toolId) {
    std::string message, structured;
    GrimToolResult result;
    try {
        const auto args = request->arguments_size ? nlohmann::json::parse(request->arguments,request->arguments+request->arguments_size) : nlohmann::json::object();
        structured = runtime->request(toolId,args).dump();
        result.success = true; result.error_code = "ERR_NONE"; result.category = "information";
        message = "Perception snapshot";
    } catch (const std::exception& e) {
        message = e.what(); result.error_code = "ERR_PERCEPTION"; result.category = "error";
    } catch (...) {
        message = "Unknown perception failure"; result.error_code = "ERR_PERCEPTION"; result.category = "error";
    }
    result.message = message.c_str(); result.structured_output_json = structured.c_str();
    return writer->complete(writer,&result);
}
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version != GRIM_API_VERSION || !api->subscribe_system_tick ||
        !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::plugin::setLoggingAPI(api);
    try {
        runtime = std::make_unique<perceptionPlugin::Runtime>(api);
        return GRIM_OK;
    } catch (...) { runtime.reset(); grim::plugin::setLoggingAPI(nullptr); return GRIM_ERROR_UNKNOWN; }
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    for (const auto* supported : ids)
        if (id == supported) return call(request, writer, supported);
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { runtime.reset(); grim::plugin::setLoggingAPI(nullptr); }
