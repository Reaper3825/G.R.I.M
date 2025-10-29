// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin
#include "pch.hpp"
#include "commands/commands_core.hpp"
#include "commands/commands_osint.hpp"
#include "core/plugin.hpp"
#include "logger.hpp"

// Wrapper functions to convert CommandResult to PluginCommandFunc signature
static CommandResult wrapProfilePerson(const std::string& args) {
    return cmdProfilePerson(args);
}

static CommandResult wrapSherlockSweep(const std::string& args) {
    return cmdSherlockSweep(args);
}

static CommandResult wrapOsintReport(const std::string& args) {
    return cmdOsintReport(args);
}

static CommandResult wrapOsintStatus(const std::string& args) {
    return cmdOsintStatus(args);
}

static CommandResult wrapOsintClearCache(const std::string& args) {
    return cmdOsintClearCache(args);
}

static CommandResult wrapOsintScanSecrets(const std::string& args) {
    return cmdOsintScanSecrets(args);
}

static CommandResult wrapOsintShowSecrets(const std::string& args) {
    return cmdOsintShowSecrets(args);
}

static CommandResult wrapOsintShowUI(const std::string& args) {
    return cmdOsintShowUI(args);
}

extern "C" GRIM_PLUGIN_API void registerGrimPlugin() {
    LOG_DEBUG("Plugin", "OSINT Self-Audit plugin v2.1.0 loading");

    // Register main OSINT commands
    grim_register_command("profile_person", wrapProfilePerson);
    grim_register_command("sherlock_sweep", wrapSherlockSweep);
    grim_register_command("osint_report", wrapOsintReport);
    
    // Register utility commands
    grim_register_command("osint_status", wrapOsintStatus);
    grim_register_command("osint_clear_cache", wrapOsintClearCache);
    grim_register_command("osint_scan_secrets", wrapOsintScanSecrets);
    grim_register_command("osint_show_secrets", wrapOsintShowSecrets);
    grim_register_command("osint_show_ui", wrapOsintShowUI);  // NEW
    
    // Aliases for convenience
    grim_register_command("osint", wrapOsintReport);  // Alias for osint_report
    grim_register_command("sherlock", wrapSherlockSweep);  // Short alias

    LOG_DEBUG("Plugin", "OSINT commands registered (8 commands + 2 aliases)");
}
