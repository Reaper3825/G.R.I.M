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
static const GrimPluginAPI* g_api = nullptr;

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
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdProfilePerson(arguments(call)));
}
static GrimResult handleSherlockSweep(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdSherlockSweep(arguments(call)));
}
static GrimResult handleOsintReport(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintReport(arguments(call)));
}
static GrimResult handleOsintStatus(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintStatus(arguments(call)));
}
static GrimResult handleOsintClearCache(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintClearCache(arguments(call, false)));
}
static GrimResult handleOsintScanSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintScanSecrets(arguments(call)));
}
static GrimResult handleOsintShowSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    std::lock_guard<std::mutex> lock(commandsMutex);
    if (!g_api || !g_api->is_provider_current(g_api) || !g_api->has_permission(g_api, GRIM_PERM_FILESYSTEM) || !g_api->has_permission(g_api, GRIM_PERM_NETWORK) || !g_api->has_permission(g_api, GRIM_PERM_PROCESS)) return GRIM_ERROR_PERMISSION_DENIED;
    return deliver(writer, cmdOsintShowSecrets(arguments(call)));
}
// ============================================================================
// NEW PLUGIN API IMPLEMENTATION
// ============================================================================

static GrimPluginInfo s_plugin_info = {
    GRIM_API_VERSION,
    "OSINT Self-Audit Plugin",
    "2.1.0",
    "GRIM Project",
    "Privacy-focused OSINT scanning and self-audit tools",
    GRIM_PERM_FILESYSTEM | GRIM_PERM_NETWORK | GRIM_PERM_PROCESS
};

extern "C" {

PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    return &s_plugin_info;
}

PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        api->api_version < GRIM_API_VERSION || !api->is_provider_current || !api->has_permission) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    osintProvider::initializePaths();
    osintProvider::cancelled = false;
    g_api = api;
    grim::plugin::setLoggingAPI(api);
    
    api->log(api, GRIM_LOG_INFO, "OSINT Self-Audit plugin v2.1.0 loading");
    
    struct Entry { const char* id; const char* description; const char* usage; GrimToolHandler handler; bool informational; bool required; };
    const Entry entries[] = {
        {"profile_person", "Profile a username across public sources", "profile_person {t} [--no-cache] [--verbose]", handleProfilePerson, false, true},
        {"sherlock_sweep", "Search public sites for a username", "sherlock_sweep {t} [--async]", handleSherlockSweep, false, true},
        {"osint_report", "Read a username report and optionally export it", "osint_report {t} [--export {o}]", handleOsintReport, false, true},
        {"osint_status", "Read scan status and consume completed results", "osint_status [{t}]", handleOsintStatus, false, false},
        {"osint_clear_cache", "Delete cached OSINT reports", "osint_clear_cache", handleOsintClearCache, false, false},
        {"osint_scan_secrets", "Scan discovered URLs for exposed sensitive data", "osint_scan_secrets {t}", handleOsintScanSecrets, false, true},
        {"osint_show_secrets", "Read discovered sensitive data for a username", "osint_show_secrets {t} [--severity <level>]", handleOsintShowSecrets, true, true}
    };
    constexpr size_t toolCount = sizeof(entries) / sizeof(entries[0]);
    GrimToolRegistration tools[toolCount];
    GrimToolParameter arguments[toolCount];
    const char* summaries[] = {"Build a public source profile for a username",
        "Search public websites for accounts matching a username",
        "Read a username report and optionally export findings",
        "Check scan status and collect completed scan results",
        "Delete locally cached reports from previous public searches",
        "Scan discovered web addresses for exposed sensitive information",
        "Read sensitive information findings for a selected username"};
    const GrimToolBinding usernameBindings[] = {{"t", "t"}, {"o", "o"}};
    const char* reportAliases[] = {"osint"};
    const char* sweepAliases[] = {"sherlock"};
    for (size_t i = 0; i < toolCount; ++i) {
        tools[i].usage_type = GRIM_TOOL_USAGE_IGNORE;
        tools[i].short_description = summaries[i];
        if (i != 4) {
            tools[i].bindings = usernameBindings; tools[i].binding_count = i == 2 ? 2 : 1;
        }
        const auto& entry = entries[i];
        tools[i].handler = entry.handler;
        auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = entry.id;
        d.description = entry.description;
        d.usage = entry.usage;
        d.version = "2.1.0";
        d.category = "osint";
        d.argument_format = "raw_text";
        d.required_permissions = s_plugin_info.required_permissions;
        d.is_informational = entry.informational;
        d.needs_confirmation = !entry.informational;
        const char* args = std::strchr(entry.usage, ' ');
        if (args) {
            arguments[i].name = "arguments";
            arguments[i].type = "string";
            arguments[i].description = args + 1;
            arguments[i].required = entry.required;
            d.parameters = &arguments[i];
            d.parameter_count = 1;
        }
    }
    tools[1].descriptor.aliases = {sweepAliases, 1};
    tools[2].descriptor.aliases = {reportAliases, 1};
    return api->register_tools(api, tools, toolCount);

}

PLUGIN_EXPORT void grim_plugin_shutdown() {
    osintProvider::cancelled = true;
    shutdownOsintScans();
    if (g_api) {
        g_api->log(g_api, GRIM_LOG_INFO, "OSINT plugin shutting down");
    }
    grim::plugin::setLoggingAPI(nullptr);
    g_api = nullptr;
}

PLUGIN_EXPORT void grim_plugin_reload() {
    if (g_api) {
        g_api->log(g_api, GRIM_LOG_INFO, "OSINT plugin reloaded");
    }
}

} // extern "C"
