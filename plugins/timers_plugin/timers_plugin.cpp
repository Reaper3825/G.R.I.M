#include "plugins/runtime_service_client.hpp"
#include "timer_provider.hpp"
#include <memory>
namespace { std::unique_ptr<grim::timers::Provider> provider; }


CommandResult cmdSetTimer(grim::timers::Provider& state, const std::string& arg) {
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

    state.add(seconds, "Timer expired after " + std::to_string(seconds) + "s");

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
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void* context) {
    return grim::runtime::deliver(writer, cmdSetTimer(*static_cast<grim::timers::Provider*>(context), std::string(request->arguments, request->arguments_size)));
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Timers Plugin", "1.0.0", "GRIM Project",
        "Plugin-owned countdown timers and queue resources", (GRIM_PERM_SYSTEM | GRIM_PERM_UI)};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->create_timer || !api->cancel_timer || !api->replace_resources || !api->show_notification) return GRIM_ERROR_API_VERSION_MISMATCH;
    provider = std::make_unique<grim::timers::Provider>(api);
    GrimToolRegistration tool;
    tool.short_description = "Start a countdown timer for the specified duration";
    tool.handler = call; tool.user_data = provider.get();
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = "set_timer"; d.version = "1.0.0";
    d.description = "Set a countdown timer"; d.usage = "set_timer <seconds>"; d.category = "action";
    d.argument_format = "raw_text"; d.required_permissions = (GRIM_PERM_SYSTEM | GRIM_PERM_UI); d.needs_confirmation = 1;
    GrimToolParameter parameter;
    parameter.name = "seconds"; parameter.type = "number";
    parameter.description = "Positive integer seconds; retains the existing integer-prefix parsing"; parameter.required = 1;
    d.parameters = &parameter; d.parameter_count = 1;
    const auto result = api->register_tools(api, &tool, 1);
    if (result != GRIM_OK) return result;
    try { provider->prepare(); return GRIM_OK; } catch (...) { return GRIM_ERROR_UNKNOWN; }
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() { provider.reset(); }
