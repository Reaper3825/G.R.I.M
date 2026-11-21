#include "core/plugin_tool_result.hpp"
#include <string_view>
// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin (NEW API)
#include "pch.hpp"
#include "commands/command_result.hpp"
#include "commands_osint.hpp"
#include "core/plugin_api.hpp"
#include "plugins/plugin_logging.hpp"
#include <cstring>
#include <mutex>
#include <sstream>
#include "provider_paths.hpp"
#include "process.hpp"
static std::mutex commandsMutex;
static std::string arguments(const GrimToolCall* call, bool target = true) {
    if (call->arguments_size > 65536 || (call->arguments_size && !call->arguments)) throw std::invalid_argument("Invalid OSINT arguments");
    std::string text = call->arguments_size ? std::string(call->arguments,call->arguments_size) : std::string{};
    if (target && !text.empty()) { std::istringstream stream(text); std::string name; stream >> name; osintProvider::validateTarget(name); }
    return text;
}

// Store the API pointer globally for use in command handlers
static grim::PluginHost g_api = {};

static GrimToolReply handleProfilePerson(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdProfilePerson(arguments(call)));
}
static GrimToolReply handleSherlockSweep(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdSherlockSweep(arguments(call)));
}
static GrimToolReply handleOsintReport(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdOsintReport(arguments(call)));
}
static GrimToolReply handleOsintStatus(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdOsintStatus(arguments(call)));
}
static GrimToolReply handleOsintClearCache(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdOsintClearCache(arguments(call, false)));
}
static GrimToolReply handleOsintScanSecrets(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdOsintScanSecrets(arguments(call)));
}
static GrimToolReply handleOsintShowSecrets(const GrimToolCall* call, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api) return {GRIM_ERROR_PLUGIN_NOT_LOADED};
    return grim::toolReply(cmdOsintShowSecrets(arguments(call)));
}
// ============================================================================
// NEW PLUGIN API IMPLEMENTATION
// ============================================================================

extern "C" {

PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    osintProvider::initializePaths();
    osintProvider::cancelled = false;
    g_api = api;
    grim::plugin::setLoggingAPI(api);
    
    api->log(api.instance(), GRIM_LOG_INFO, "OSINT Self-Audit plugin v2.1.0 loading");
    
    return GRIM_OK;
}

PLUGIN_EXPORT GrimToolReply grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request) {
    if (!tool_id) return {GRIM_ERROR_INVALID_PARAM};
    const std::string_view id(tool_id);
    if (id == "profile_person") return handleProfilePerson(request, nullptr);
    if (id == "sherlock_sweep") return handleSherlockSweep(request, nullptr);
    if (id == "osint_report") return handleOsintReport(request, nullptr);
    if (id == "osint_status") return handleOsintStatus(request, nullptr);
    if (id == "osint_clear_cache") return handleOsintClearCache(request, nullptr);
    if (id == "osint_scan_secrets") return handleOsintScanSecrets(request, nullptr);
    if (id == "osint_show_secrets") return handleOsintShowSecrets(request, nullptr);
    return {GRIM_ERROR_NOT_FOUND};
}

PLUGIN_EXPORT void grim_plugin_shutdown() {
    osintProvider::cancelled = true;
    shutdownOsintScans();
    if (g_api) {
        g_api->log(g_api.instance(), GRIM_LOG_INFO, "OSINT plugin shutting down");
    }
    grim::plugin::setLoggingAPI({});
    g_api = {};
}

} // extern "C"
