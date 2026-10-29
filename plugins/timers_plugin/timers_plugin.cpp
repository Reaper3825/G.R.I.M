#include "plugins/runtime_service_client.hpp"


CommandResult cmdSetTimer(const std::string& arg) {
    int seconds = 0;
    try {
        seconds = std::stoi(arg);
    } catch (...) {
        return {
            false,                                          // success
            "[Timer][Error] Invalid number of seconds.",   // message
            "ERR_TIMER_INVALID",                            // errorCode
            "error",                                        // category
            "Invalid timer duration",                       // voice
            Colors::Red                                     // color
        };
    }

    if (seconds <= 0) {
        return {
            false,                                              // success
            "[Timer][Error] Duration must be positive.",        // message
            "ERR_TIMER_NONPOSITIVE",                            // errorCode
            "error",                                            // category
            "Non-positive timer duration",                      // voice
            Colors::Red                                         // color
        };
    }

    grim::runtime::response(grim::runtime::host->schedule_user_timer(grim::runtime::host, seconds,
        ("Timer expired after " + std::to_string(seconds) + "s").c_str()));

    return {
        true,                                                   // success
        "[Timer] Set for " + std::to_string(seconds) + " seconds.", // message
        "ERR_NONE",                                             // errorCode
        "routine",                                              // category
        "Timer set",                                            // voice
        Colors::Green                                           // color
    };
}

namespace {
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    return grim::runtime::deliver(writer, cmdSetTimer(std::string(request->arguments, request->arguments_size)));
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Timers Plugin", "1.0.0", "GRIM Project",
        "Countdown timer command using the host-owned timer queue", GRIM_PERM_SYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->schedule_user_timer) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::runtime::host = api;
    GrimToolRegistration tool;
    tool.handler = call;
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = "set_timer"; d.version = "1.0.0";
    d.description = "Set a countdown timer"; d.usage = "set_timer <seconds>"; d.category = "action";
    d.argument_format = "raw_text"; d.required_permissions = GRIM_PERM_SYSTEM; d.needs_confirmation = 1;
    GrimToolParameter parameter;
    parameter.name = "seconds"; parameter.type = "number";
    parameter.description = "Positive integer seconds; retains the existing integer-prefix parsing"; parameter.required = 1;
    d.parameters = &parameter; d.parameter_count = 1;
    const char* examples[] = {"set_timer 60"}; d.examples = {examples, 1};
    return api->register_tools(api, &tool, 1);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { grim::runtime::host = nullptr; }
