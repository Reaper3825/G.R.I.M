#include "core/plugin_api.hpp"
#include "ui_training_panel.hpp"
#include <memory>

namespace {
std::unique_ptr<UITrainingPanel> workspace;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || api->api_version < GRIM_API_VERSION ||
        (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || !api->log)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        workspace = std::make_unique<UITrainingPanel>();
        workspace->attachToProvider(instance);
        return GRIM_OK;
    } catch (const std::exception& error) {
        api->log(api.instance(), GRIM_LOG_ERROR, error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    // The host has already detached this generation from drawing/input.
    workspace.reset();
}
