#include "core/plugin_tool_registration.hpp"
#include "core/plugin_api.hpp"
#include <string>

#ifndef FIXTURE_NAME
#define FIXTURE_NAME "Shutdown fixture"
#endif
#ifndef FIXTURE_API_VERSION
#define FIXTURE_API_VERSION GRIM_API_VERSION
#endif

static const GrimPluginAPI* api = nullptr;
#ifdef FIXTURE_NATIVE_DEPENDENCY
extern "C" __declspec(dllimport) int grim_test_dependency();
#endif
#ifdef FIXTURE_TOOLS
#ifndef FIXTURE_RESULT
#define FIXTURE_RESULT "fixture result"
#endif
static void event(const char*, const void*, size_t, void*) {
    api->set_context(api, "fixture_event", FIXTURE_RESULT);
}
static void tick(double, double, void*) {
    api->set_context(api, "fixture_tick", FIXTURE_RESULT);
}
static GrimResult run(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    if (std::string(call->arguments, call->arguments_size) == "reload") api->request_reload(api);
    GrimToolResult result;
    result.success = 1;
    result.message = FIXTURE_RESULT;
    return writer->complete(writer, &result);
}
#endif
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{FIXTURE_API_VERSION, FIXTURE_NAME, "1.0", "tests", "Shutdown lifecycle fixture", 0};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* value) {
    api = value;
#ifdef FIXTURE_NATIVE_DEPENDENCY
    if (grim_test_dependency() != 42) return GRIM_ERROR_UNKNOWN;
#endif
#ifdef FIXTURE_TOOLS
    GRIM::PluginToolRegistration tool;
    tool.handler = run;
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = FIXTURE_NAME;
    d.description = "Fixture tool";
    d.version = "1.0";
    d.usage = FIXTURE_NAME;
    d.category = "test";
    d.argument_format = "raw_text";
    d.is_informational = false;
#ifdef FIXTURE_COLLISION
    const char* aliases[] = {"reserved_fixture_tool"};
    d.aliases.assign(aliases, aliases + 1);
#endif
    if (GRIM::registerPluginTools(api, &tool, 1) != GRIM_OK) return GRIM_ERROR_INVALID_PARAM;
#ifdef FIXTURE_EXTRA
    tool.descriptor.tool_id = tool.descriptor.display_name = FIXTURE_NAME "-extra";
    if (GRIM::registerPluginTools(api, &tool, 1) != GRIM_OK) return GRIM_ERROR_INVALID_PARAM;
#endif
    api->subscribe_event(api, "fixture_event", event, nullptr);
    if (const auto result = api->subscribe_system_tick(api, tick, nullptr); result != GRIM_OK)
        return result;
#endif
#ifdef FIXTURE_FAIL_INIT
    return GRIM_ERROR_UNKNOWN;
#endif
    return GRIM_OK;
}
#ifdef FIXTURE_ACTIVATION
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_activate(const GrimPluginAPI* value) {
    if (!value->is_provider_current(value)) return GRIM_ERROR_UNKNOWN;
#ifdef FIXTURE_FAIL_ACTIVATION
    return GRIM_ERROR_UNKNOWN;
#else
    const auto* previous = value->get_context(value, "fixture_retired");
    const std::string retired = previous ? previous : "";
    value->string_free(value, previous);
    return value->set_context(value, "fixture_activated", (std::string(FIXTURE_RESULT) + ":" + retired).c_str());
#endif
}
#endif

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
#ifdef FIXTURE_ACTIVATION
    api->set_context(api, "fixture_retired", FIXTURE_RESULT);
#endif
    api->log(api, GRIM_LOG_INFO, FIXTURE_NAME);
}
