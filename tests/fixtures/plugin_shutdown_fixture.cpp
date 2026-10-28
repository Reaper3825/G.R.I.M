#include "core/plugin_api.hpp"

#ifndef FIXTURE_NAME
#define FIXTURE_NAME "Shutdown fixture"
#endif
#ifndef FIXTURE_API_VERSION
#define FIXTURE_API_VERSION GRIM_API_VERSION
#endif

static const GrimPluginAPI* api = nullptr;
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{FIXTURE_API_VERSION, FIXTURE_NAME, "1.0", "tests", "Shutdown lifecycle fixture", 0};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* value) {
    api = value;
#ifdef FIXTURE_FAIL_INIT
    return GRIM_ERROR_UNKNOWN;
#endif
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    api->log(api, GRIM_LOG_INFO, FIXTURE_NAME);
}
