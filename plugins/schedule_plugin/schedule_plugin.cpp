#include <string_view>
#include "core/plugin_api.hpp"
#include "schedule_provider.hpp"
#include "calendar_panel.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <array>
namespace {
std::unique_ptr<grim::scheduling::Provider> provider;
std::unique_ptr<grim::scheduling::CalendarPanel> calendarPanel;
const char* operations[]={"create","get","list","update","cancel","status"};
GrimResult call(const GrimToolCall* request,GrimToolResultWriter* writer,const char* operation) {
    if(!writer || !writer->complete) return GRIM_ERROR_INVALID_PARAM;
    nlohmann::json value;
    try {
        if(!provider || !request || !request->arguments || request->arguments_size>65536) throw std::runtime_error("Invalid scheduling request");
        value=provider->execute(operation,
            nlohmann::json::parse(request->arguments,request->arguments+request->arguments_size));
    } catch(const std::exception& error) {value={{"success",false},{"error","service_error"},{"message",error.what()}};}
    const std::string output=value.dump();
    const bool success=value.value("success",false);
    const std::string message=success?"Scheduling operation completed":value.value("message",std::string("Scheduling operation failed"));
    const std::string code=success?"ERR_NONE":value.value("error",std::string("service_error"));
    GrimToolResult result;result.success=success;result.message=message.c_str();result.error_code=code.c_str();
    result.category=success?"schedule":"error";result.structured_output_json=output.c_str();result.color_rgb=success?0x66CC99:0xFF6464;
    return writer->complete(writer,&result);
}
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if(!api || (api->api_version>>16)!=GRIM_API_VERSION_MAJOR || api->api_version<GRIM_API_VERSION ||
        !api->subscribe_system_tick || !api->replace_resources || !api->show_notification ||
        !api->string_free || !api->log) return GRIM_ERROR_API_VERSION_MISMATCH;
    provider=std::make_unique<grim::scheduling::Provider>(api);
    try {
        provider->prepare();
        calendarPanel = std::make_unique<grim::scheduling::CalendarPanel>(api, *provider);
        calendarPanel->attachToProvider(instance);
        return GRIM_OK;
    } catch(const std::exception& error) {
        api->log(api.instance(), GRIM_LOG_ERROR, error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    constexpr std::string_view prefix = "schedule.";
    if (id.substr(0, prefix.size()) == prefix) {
        const auto operation = id.substr(prefix.size());
        for (const auto* supported : operations)
            if (operation == supported) return call(request, writer, supported);
    }
    return GRIM_ERROR_NOT_FOUND;
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown(){calendarPanel.reset();provider.reset();}
