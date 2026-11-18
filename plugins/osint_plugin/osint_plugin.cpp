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

// The writer copies while the local CommandResult is still alive.
static GrimResult deliver(GrimToolResultWriter* writer, const CommandResult& value) {
    GrimToolResult result;
    result.success = value.success;
    result.message = value.message.c_str();
    result.error_code = value.errorCode.c_str();
    result.category = value.category.c_str();
    result.voice = value.voice.c_str();
    result.structured_output_json = value.structuredOutputJson.c_str();
    result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
    return writer->complete(writer, &result);
}
static GrimResult handleProfilePerson(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdProfilePerson(arguments(call)));
}
static GrimResult handleSherlockSweep(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdSherlockSweep(arguments(call)));
}
static GrimResult handleOsintReport(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintReport(arguments(call)));
}
static GrimResult handleOsintStatus(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintStatus(arguments(call)));
}
static GrimResult handleOsintClearCache(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintClearCache(arguments(call, false)));
}
static GrimResult handleOsintScanSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintScanSecrets(arguments(call)));
}
static GrimResult handleOsintShowSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->has_permission(g_api.instance(), GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api.instance(), GRIM_PERM_NETWORK) || !g_api->has_permission(g_api.instance(), GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintShowSecrets(arguments(call)));
}
// ============================================================================
// NEW PLUGIN API IMPLEMENTATION
// ============================================================================

extern "C" {

PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        api->api_version < GRIM_API_VERSION || !api->has_permission) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    osintProvider::initializePaths();
    osintProvider::cancelled = false;
    g_api = api;
    grim::plugin::setLoggingAPI(api);
    
    api->log(api.instance(), GRIM_LOG_INFO, "OSINT Self-Audit plugin v2.1.0 loading");
    
    return GRIM_OK;
}

PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id) return GRIM_ERROR_INVALID_PARAM;
    const std::string_view id(tool_id);
    if (id == "profile_person") return handleProfilePerson(request, writer, nullptr);
    if (id == "sherlock_sweep") return handleSherlockSweep(request, writer, nullptr);
    if (id == "osint_report") return handleOsintReport(request, writer, nullptr);
    if (id == "osint_status") return handleOsintStatus(request, writer, nullptr);
    if (id == "osint_clear_cache") return handleOsintClearCache(request, writer, nullptr);
    if (id == "osint_scan_secrets") return handleOsintScanSecrets(request, writer, nullptr);
    if (id == "osint_show_secrets") return handleOsintShowSecrets(request, writer, nullptr);
    return GRIM_ERROR_NOT_FOUND;
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
