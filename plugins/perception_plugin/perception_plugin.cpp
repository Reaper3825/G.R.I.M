#include "perception_runtime.hpp"
#include "plugins/plugin_logging.hpp"
#include <memory>
namespace {
std::unique_ptr<perceptionPlugin::Runtime> runtime;
const char* ids[] = {"perception.observe", "perception.digital.start", "perception.digital.stop",
    "perception.physical.open", "perception.physical.close", "perception.configure", "perception.calibrate"};
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void* context) {
    std::string message, structured;
    GrimToolResult result;
    try {
        const auto args = request->arguments_size ? nlohmann::json::parse(request->arguments,request->arguments+request->arguments_size) : nlohmann::json::object();
        structured = runtime->request(static_cast<const char*>(context),args).dump();
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
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION,"Perception Plugin","2.0.0","GRIM Project",
        "Plugin-owned digital and physical perception primitives",GRIM_PERM_SYSTEM | GRIM_PERM_FILESYSTEM | GRIM_PERM_NETWORK};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version != GRIM_API_VERSION || !api->register_tools || !api->create_timer ||
        !api->cancel_timer || !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::plugin::setLoggingAPI(api);
    try {
        runtime = std::make_unique<perceptionPlugin::Runtime>(api);
        GrimToolRegistration registrations[7];
        const char* descriptions[] = {"Read latest structured perception observations", "Start digital capture and primitives",
            "Stop digital capture and primitives", "Open a physical camera source", "Close physical capture and processing",
            "Configure a plugin-owned perception operator", "Begin physical camera calibration"};
        for (size_t i=0;i<7;++i) {
            auto& r=registrations[i]; auto& d=r.descriptor;
            r.handler=call; r.user_data=const_cast<char*>(ids[i]); r.usage_type=GRIM_TOOL_USAGE_DEBUG;
            r.short_description=descriptions[i]; d.tool_id=d.display_name=d.usage=ids[i]; d.version="2.0.0";
            d.description=descriptions[i]; d.category="perception"; d.argument_format="json";
            d.argument_schema_json=R"({"type":"object"})";
            d.required_permissions=GRIM_PERM_SYSTEM | (i==0 ? 0 : GRIM_PERM_FILESYSTEM | GRIM_PERM_NETWORK);
            d.is_informational=i==0; d.needs_confirmation=i!=0;
        }
        return api->register_tools(api,registrations,7);
    } catch (...) { runtime.reset(); grim::plugin::setLoggingAPI(nullptr); return GRIM_ERROR_UNKNOWN; }
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { runtime.reset(); grim::plugin::setLoggingAPI(nullptr); }
