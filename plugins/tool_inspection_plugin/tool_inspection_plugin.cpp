#include "snapshot_registry.hpp"
#include "plugins/plugin_logging.hpp"

namespace {
template<CommandResult (*Handler)(const std::string&)>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    const auto value = Handler(std::string(request->arguments, request->arguments_size));
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
    static const GrimPluginInfo info{GRIM_API_VERSION, "Tool Inspection Plugin", "1.0.0",
        "GRIM Project", "Read-only registry tool listing, documentation, and usage statistics", 0};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_tool_snapshot) return GRIM_ERROR_API_VERSION_MISMATCH;
    inspection::host = api;
    grim::plugin::setLoggingAPI(api);
    GrimToolRegistration tools[3];
    tools[0].handler = call<cmdListTools>;
    tools[1].handler = call<cmdToolInfo>;
    tools[2].handler = call<cmdToolStats>;
    const char* ids[] = {"list_tools", "tool_info", "tool_stats"};
    const char* descriptions[] = {"List all registered commands/tools", "Show detailed information about a tool", "Show tool usage statistics"};
    const char* usages[] = {"list_tools [category]", "tool_info <command_name>", "tool_stats [limit]"};
    const char* examples[] = {"list_tools information", "tool_info list_tools", "tool_stats 10"};
    const char* names[] = {"category", "command_name", "limit"};
    const char* parameterDescriptions[] = {"Exact category; omitted lists all tools", "Canonical tool ID, display name, or registered alias", "Maximum rows; defaults to 20; retains existing integer-prefix parsing"};
    GrimToolParameter parameters[3];
    for (size_t i = 0; i < 3; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i];
        d.version = "1.0.0";
        d.description = descriptions[i];
        d.usage = usages[i];
        d.category = "information";
        d.argument_format = "raw_text";
        d.is_informational = 1;
        d.examples = {&examples[i], 1};
        parameters[i].name = names[i];
        parameters[i].type = i == 2 ? "number" : "string";
        parameters[i].description = parameterDescriptions[i];
        parameters[i].required = i == 1;
        d.parameters = &parameters[i];
        d.parameter_count = 1;
    }
    return api->register_tools(api, tools, 3);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI(nullptr);
    inspection::host = nullptr;
}
