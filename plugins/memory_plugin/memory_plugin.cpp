#include "host_memory.hpp"
#include "plugins/plugin_logging.hpp"
namespace {
template<CommandResult (*Handler)(const std::string&)>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    return grim::runtime::deliver(writer, Handler(std::string(request->arguments, request->arguments_size)));
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Memory Plugin", "1.0.0", "GRIM Project",
        "Manual memory commands using the host rotation pipeline and storage", GRIM_PERM_MEMORY};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->stage_manual_memory || !api->search_memory || !api->flush_memory_record_if_present)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::runtime::host = api;
    grim::plugin::setLoggingAPI(api);
    GrimToolRegistration tools[3];
    tools[0].handler = call<cmdRemember>; tools[1].handler = call<cmdRecall>; tools[2].handler = call<cmdForget>;
    const char* ids[] = {"remember", "recall", "forget"};
    const char* descriptions[] = {"Stage manual information in the host memory rotation pipeline", "Retrieve information from memory",
        "Request legacy forget behavior: check and flush matching records without deleting them"};
    const char* usages[] = {"remember <text>", "recall <query>", "forget <query>"};
    const char* examples[] = {"remember my color is blue", "recall color", "forget color"};
    GrimToolParameter parameters[3];
    for (size_t i = 0; i < 3; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0";
        d.description = descriptions[i]; d.usage = usages[i];
        d.category = i == 1 ? "information" : "action"; d.argument_format = "raw_text";
        d.required_permissions = GRIM_PERM_MEMORY; d.is_informational = i == 1; d.needs_confirmation = i != 1;
        d.examples = {&examples[i], 1};
        parameters[i].name = i == 0 ? "text" : "query"; parameters[i].type = "string";
        parameters[i].description = i == 0 ? "Original manual memory text staged in the host rotation pipeline" :
            "Unmodified query for the existing host memory search";
        parameters[i].required = 1; d.parameters = &parameters[i]; d.parameter_count = 1;
    }
    return api->register_tools(api, tools, 3);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI(nullptr); grim::runtime::host = nullptr;
}
