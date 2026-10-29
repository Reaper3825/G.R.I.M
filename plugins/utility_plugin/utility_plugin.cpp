// Fixed-response utility commands.
#include "core/plugin_api.hpp"

namespace {
GrimResult clean(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    GrimToolResult result;
    result.success = 1;
    result.message = "[Utility] Console cleared.";
    result.error_code = "ERR_NONE";
    result.category = "routine";
    result.voice = "Console cleared";
    result.color_rgb = 0x64FF64; // Colors::Green
    return writer->complete(writer, &result);
}

GrimResult help(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    // Static legacy help excludes retired agent/admin commands.
    // Dynamic registry help is available separately through list_tools/tool_info.
    GrimToolResult result;
    result.success = 1;
    result.message =
        "[Help] Available commands:\n"
        "- remember <key> <value>\n"
        "- recall <key>\n"
        "- forget <key>\n"
        "- pwd\n"
        "- cd <dir>\n"
        "- ls\n"
        "- mkdir <dir>\n"
        "- rm <file>\n"
        "- set_timer <seconds>\n"
        "- sysinfo\n"
        "- clean\n"
        "- help\n"
        "- voice\n"
        "- voice_stream\n";
    result.error_code = "ERR_NONE";
    result.category = "summary";
    result.voice = "Help shown";
    result.color_rgb = 0x64FFFF; // Colors::Cyan
    return writer->complete(writer, &result);
}
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Utility Plugin", "1.0.0",
        "GRIM Project", "Existing console utility responses and static command help", 0};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    GrimToolRegistration tools[2];
    tools[0].handler = clean;
    tools[1].handler = help;
    const char* ids[] = {"clean", "help"};
    const char* descriptions[] = {"Clean console history", "Show available commands"};
    const char* categories[] = {"system", "information"};
    for (size_t i = 0; i < 2; ++i) {
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = d.usage = ids[i];
        d.version = "1.0.0";
        d.description = descriptions[i];
        d.category = categories[i];
        d.argument_format = "raw_text";
        d.is_informational = i == 1;
        d.needs_confirmation = i == 0;
        d.examples = {&ids[i], 1};
    }
    return api->register_tools(api, tools, 2);
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}
