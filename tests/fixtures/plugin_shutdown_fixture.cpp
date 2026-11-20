#include "core/plugin_api.hpp"
#include <string>
#include <string_view>

#ifndef FIXTURE_NAME
#define FIXTURE_NAME "Shutdown fixture"
#endif

static grim::PluginHost api;
#ifdef FIXTURE_NATIVE_DEPENDENCY
extern "C" __declspec(dllimport) int grim_test_dependency();
#endif
#ifdef FIXTURE_TOOLS
#ifndef FIXTURE_RESULT
#define FIXTURE_RESULT "fixture result"
#endif
static GrimResult publish(const char* kind) {
    const auto json = std::string("[{\"resource_id\":\"") + kind +
        "\",\"kind\":\"" + kind + "\",\"name\":\"Fixture\",\"runtime_id\":1,\"state\":{\"value\":\"" +
        FIXTURE_RESULT + "\"}}]";
    return api->replace_resources(api.instance(), kind, json.data(), json.size());
}
static void tick(double, double, void*) { publish("fixture_tick"); }
static GrimResult run(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    if (std::string(call->arguments, call->arguments_size) == "reload") api->request_reload(api.instance());
    GrimToolResult result;
    result.success = 1;
    result.message = FIXTURE_RESULT;
    return writer->complete(writer, &result);
}
#endif
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* value, GrimPluginHandle instance) {
    if (!value || value->api_version != GRIM_API_VERSION) return GRIM_ERROR_API_VERSION_MISMATCH;
    api = {value, instance};
#ifdef FIXTURE_NATIVE_DEPENDENCY
    if (grim_test_dependency() != 42) return GRIM_ERROR_UNKNOWN;
#endif
#ifdef FIXTURE_TOOLS
    api->log(instance, GRIM_LOG_INFO, "init " FIXTURE_RESULT);
    if (publish("fixture_init") != GRIM_OK) return GRIM_ERROR_UNKNOWN;
    if (const auto result = api->subscribe_system_tick(instance, tick, nullptr); result != GRIM_OK)
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
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    api->log(api.instance(), GRIM_LOG_INFO, FIXTURE_NAME);
    api = {};
}
