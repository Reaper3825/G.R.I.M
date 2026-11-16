#include "core/plugin_api.hpp"
#include <string>
#include <string_view>

#ifndef FIXTURE_NAME
#define FIXTURE_NAME "Shutdown fixture"
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
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* value) {
    if (!value || value->api_version != GRIM_API_VERSION) return GRIM_ERROR_API_VERSION_MISMATCH;
    api = value;
#ifdef FIXTURE_NATIVE_DEPENDENCY
    if (grim_test_dependency() != 42) return GRIM_ERROR_UNKNOWN;
#endif
#ifdef FIXTURE_TOOLS
    api->subscribe_event(api, "fixture_event", event, nullptr);
    if (const auto result = api->subscribe_system_tick(api, tick, nullptr); result != GRIM_OK)
        return result;
#endif
#ifdef FIXTURE_FAIL_INIT
    return GRIM_ERROR_UNKNOWN;
#endif
    return GRIM_OK;
}
#ifdef FIXTURE_TOOLS
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* call, GrimToolResultWriter* writer) {
    const std::string_view id(tool_id);
    if (id == FIXTURE_NAME) return run(call, writer, nullptr);
#ifdef FIXTURE_EXTRA
    if (id == FIXTURE_NAME "-extra") return run(call, writer, nullptr);
#endif
    return GRIM_ERROR_NOT_FOUND;
}
#endif
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
