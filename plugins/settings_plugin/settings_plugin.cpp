#include "core/plugin_api.hpp"
#include "ui_settings_menu.hpp"
#include <memory>

namespace {
std::unique_ptr<UISettingsMenu> settings;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || api->api_version < GRIM_API_VERSION ||
        (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || !api->log)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    try {
        settings = std::make_unique<UISettingsMenu>(UIPanel::providerInstanceId(api));
        return GRIM_OK;
    } catch (const std::exception& error) {
        api->log(api, GRIM_LOG_ERROR, error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    // The host has already detached this generation from drawing/input.
    settings.reset();
}
