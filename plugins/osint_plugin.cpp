// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin (NEW API)
#include "pch.hpp"
#include "commands/commands_core.hpp"
#include "commands/commands_osint.hpp"
#include "core/plugin_api.hpp"
#include "logger.hpp"

// Store the API pointer globally for use in command handlers
static const GrimPluginAPI* g_api = nullptr;

// Convert CommandResult to GrimCommandResult
static GrimCommandResult toGrimResult(const CommandResult& result) {
    GrimCommandResult gr = {};
    gr.success = result.success;
    gr.message = result.message.c_str();
    gr.error_code = result.errorCode.c_str();  // Fixed: errorCode not error_code
    gr.category = result.category.c_str();
    gr.color_rgb = result.color.toUInt();  // Fixed: convert Color to uint32_t
    return gr;
}

// New API command handlers
static GrimCommandResult handleProfilePerson(const char* input, void* user_data) {
    auto result = cmdProfilePerson(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleSherlockSweep(const char* input, void* user_data) {
    auto result = cmdSherlockSweep(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintReport(const char* input, void* user_data) {
    auto result = cmdOsintReport(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintStatus(const char* input, void* user_data) {
    auto result = cmdOsintStatus(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintClearCache(const char* input, void* user_data) {
    auto result = cmdOsintClearCache(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintScanSecrets(const char* input, void* user_data) {
    auto result = cmdOsintScanSecrets(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintShowSecrets(const char* input, void* user_data) {
    auto result = cmdOsintShowSecrets(input ? input : "");
    return toGrimResult(result);
}

static GrimCommandResult handleOsintShowUI(const char* input, void* user_data) {
    auto result = cmdOsintShowUI(input ? input : "");
    return toGrimResult(result);
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
    g_api = api;
    
    api->log(GRIM_LOG_INFO, "OSINT Self-Audit plugin v2.1.0 loading");
    
    // Register all OSINT commands using new API
    api->register_command("profile_person", handleProfilePerson, nullptr);
    api->register_command("sherlock_sweep", handleSherlockSweep, nullptr);
    api->register_command("osint_report", handleOsintReport, nullptr);
    api->register_command("osint_status", handleOsintStatus, nullptr);
    api->register_command("osint_clear_cache", handleOsintClearCache, nullptr);
    api->register_command("osint_scan_secrets", handleOsintScanSecrets, nullptr);
    api->register_command("osint_show_secrets", handleOsintShowSecrets, nullptr);
    api->register_command("osint_show_ui", handleOsintShowUI, nullptr);
    
    // Register aliases
    api->register_command("osint", handleOsintReport, nullptr);
    api->register_command("sherlock", handleSherlockSweep, nullptr);
    
    api->log(GRIM_LOG_INFO, "OSINT commands registered (8 commands + 2 aliases)");
    
    return GRIM_OK;
}

PLUGIN_EXPORT void grim_plugin_shutdown() {
    if (g_api) {
        g_api->log(GRIM_LOG_INFO, "OSINT plugin shutting down");
    }
    g_api = nullptr;
}

PLUGIN_EXPORT void grim_plugin_reload() {
    if (g_api) {
        g_api->log(GRIM_LOG_INFO, "OSINT plugin reloaded");
    }
}

} // extern "C"
