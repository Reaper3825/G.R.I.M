#include "core/plugin_api.hpp"
#include "console_panel.hpp"
#include <memory>

namespace {
std::unique_ptr<ConsolePanel> console;
}

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Console Plugin", "1.0.0", "GRIM Project",
        "Own the native conversation console panel", GRIM_PERM_UI};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version < GRIM_API_VERSION ||
        (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || !api->log)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        console = std::make_unique<ConsolePanel>();
        console->setVisible(false);
        return GRIM_OK;
    } catch (const std::exception& error) {
        api->log(api, GRIM_LOG_ERROR, error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    // The host has already detached this generation from drawing/input.
    console.reset();
}
