// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin (NEW API)
#include "pch.hpp"
#include "commands/command_result.hpp"
#include "commands/commands_osint.hpp"
#include "core/plugin_api.hpp"
#include "plugin_logging.hpp"
#include <cstring>

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
    return deliver(writer, cmdProfilePerson(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleSherlockSweep(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdSherlockSweep(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintReport(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdOsintReport(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintStatus(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdOsintStatus(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintClearCache(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdOsintClearCache(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintScanSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdOsintScanSecrets(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintShowSecrets(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    return deliver(writer, cmdOsintShowSecrets(std::string(call->arguments, call->arguments_size)));
}
static GrimResult handleOsintShowUI(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    const auto command = std::string("ui.show_osint_results ") + std::string(call->arguments, call->arguments_size);
    return g_api->execute_command(g_api, command.c_str(), writer);
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
    GRIM_PERM_FILESYSTEM | GRIM_PERM_NETWORK | GRIM_PERM_UI | GRIM_PERM_PROCESS
};

extern "C" {

PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    return &s_plugin_info;
}

PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        api->api_version < GRIM_API_VERSION) {
        return GRIM_ERROR_API_VERSION_MISMATCH;
    }
    g_api = api;
    grim::plugin::setLoggingAPI(api);
    
    api->log(api, GRIM_LOG_INFO, "OSINT Self-Audit plugin v2.1.0 loading");
    
    struct Entry { const char* id; const char* description; const char* usage; GrimToolHandler handler; bool informational; bool required; };
    const Entry entries[] = {
        {"profile_person", "Profile a username across public sources", "profile_person <username> [--no-cache] [--verbose]", handleProfilePerson, false, true},
        {"sherlock_sweep", "Search public sites for a username", "sherlock_sweep <username> [--async]", handleSherlockSweep, false, true},
        {"osint_report", "Read a username report and optionally export it", "osint_report <username> [--export <file>]", handleOsintReport, false, true},
        {"osint_status", "Read scan status and consume completed results", "osint_status [username]", handleOsintStatus, false, false},
        {"osint_clear_cache", "Delete cached OSINT reports", "osint_clear_cache", handleOsintClearCache, false, false},
        {"osint_scan_secrets", "Scan discovered URLs for exposed sensitive data", "osint_scan_secrets <username>", handleOsintScanSecrets, false, true},
        {"osint_show_secrets", "Read discovered sensitive data for a username", "osint_show_secrets <username> [--severity <level>]", handleOsintShowSecrets, true, true},
        {"osint_show_ui", "Show the OSINT results interface", "osint_show_ui <username>", handleOsintShowUI, false, true}
    };
    GrimToolRegistration tools[8];
    GrimToolParameter arguments[8];
    const char* reportAliases[] = {"osint"};
    const char* sweepAliases[] = {"sherlock"};
    for (size_t i = 0; i < 8; ++i) {
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
    return api->register_tools(api, tools, 8);

}

PLUGIN_EXPORT void grim_plugin_shutdown() {
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
