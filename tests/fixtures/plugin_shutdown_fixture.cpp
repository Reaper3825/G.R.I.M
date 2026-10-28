#include "core/plugin_api.hpp"
#include <string>

#ifndef FIXTURE_NAME
#define FIXTURE_NAME "Shutdown fixture"
#endif
#ifndef FIXTURE_API_VERSION
#define FIXTURE_API_VERSION GRIM_API_VERSION
#endif

static const GrimPluginAPI* api = nullptr;
#ifdef FIXTURE_TOOLS
#ifndef FIXTURE_RESULT
#define FIXTURE_RESULT "fixture result"
#endif
static void event(const char*, const void*, size_t, void*) {
    api->set_context(api, "fixture_event", FIXTURE_RESULT);
}
static void timer(uint64_t, void*) {
    api->set_context(api, "fixture_timer", FIXTURE_RESULT);
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
#ifdef FIXTURE_TOOLS
    GrimToolRegistration tool;
    tool.handler = run;
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = FIXTURE_NAME;
    d.description = "Fixture tool";
    d.version = "1.0";
    d.usage = FIXTURE_NAME;
    d.category = "test";
    d.argument_format = "raw_text";
#ifdef FIXTURE_COLLISION
    const char* aliases[] = {"builtin"};
    d.aliases = {aliases, 1};
#endif
    if (api->register_tools(api, &tool, 1) != GRIM_OK) return GRIM_ERROR_INVALID_PARAM;
#ifdef FIXTURE_EXTRA
    tool.descriptor.tool_id = tool.descriptor.display_name = FIXTURE_NAME "-extra";
    if (api->register_tools(api, &tool, 1) != GRIM_OK) return GRIM_ERROR_INVALID_PARAM;
#endif
    api->subscribe_event(api, "fixture_event", event, nullptr);
    api->create_timer(api, 0, false, timer, nullptr);
#endif
#ifdef FIXTURE_FAIL_INIT
    return GRIM_ERROR_UNKNOWN;
#endif
    return GRIM_OK;
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    api->log(api, GRIM_LOG_INFO, FIXTURE_NAME);
}
