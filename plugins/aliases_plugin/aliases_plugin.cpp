#include "host_aliases.hpp"

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
    static const GrimPluginInfo info{GRIM_API_VERSION, "Aliases Plugin", "1.0.0", "GRIM Project",
        "Inspect and refresh the host's shared application aliases", GRIM_PERM_FILESYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_alias_snapshot || !api->refresh_aliases) return GRIM_ERROR_API_VERSION_MISMATCH;
    hostAliases::host = api;
    GrimToolRegistration tools[3];
    tools[0].handler = call<cmdAliasList>;
    tools[1].handler = call<cmdAliasInfo>;
    tools[2].handler = call<cmdAliasRefresh>;
    const char* ids[] = {"alias list", "alias info", "alias refresh"};
    const char* descriptions[] = {"List all defined command aliases", "Show information about a specific alias", "Refresh and reload aliases from disk"};
    const char* usages[] = {"alias list", "alias info <name>", "alias refresh"};
    const char* examples[] = {"alias list", "alias info editor", "alias refresh"};
    GrimToolParameter name;
    name.name = "name";
    name.type = "string";
    name.description = "Alias name passed unchanged to host lookup; host handles case normalization";
    name.required = 1;
    for (size_t i = 0; i < 3; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i];
        d.version = "1.0.0";
        d.description = descriptions[i];
        d.usage = usages[i];
        d.category = i == 2 ? "system" : "information";
        d.argument_format = "raw_text";
        d.is_informational = i != 2;
        d.needs_confirmation = i == 2;
        d.required_permissions = i == 2 ? GRIM_PERM_FILESYSTEM : 0;
        d.examples = {&examples[i], 1};
    }
    tools[1].descriptor.parameters = &name;
    tools[1].descriptor.parameter_count = 1;
    return api->register_tools(api, tools, 3);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { hostAliases::host = nullptr; }
