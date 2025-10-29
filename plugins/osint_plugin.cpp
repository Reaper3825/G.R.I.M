// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin
#include "pch.hpp"
#include "commands/commands_core.hpp"
#include "commands/commands_osint.hpp"
#include "core/plugin.hpp"
#include "logger.hpp"

// Wrapper functions to convert CommandResult to PluginCommandFunc signature
static CommandResult wrapProfileSelf(const std::string& args) {
    return cmdProfileSelf(args);
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

extern "C" GRIM_PLUGIN_API void registerGrimPlugin() {
    LOG_DEBUG("Plugin", "OSINT Self-Audit plugin v2.0.0 loading");

    // Register main OSINT commands
    grim_register_command("profile_self", wrapProfileSelf);
    grim_register_command("sherlock_sweep", wrapSherlockSweep);
    grim_register_command("osint_report", wrapOsintReport);
    
    // Register utility commands
    grim_register_command("osint_status", wrapOsintStatus);
    grim_register_command("osint_clear_cache", wrapOsintClearCache);
    
    // Aliases for convenience
    grim_register_command("osint", wrapOsintReport);  // Alias for osint_report
    grim_register_command("sherlock", wrapSherlockSweep);  // Short alias

    LOG_DEBUG("Plugin", "OSINT commands registered (5 commands + 2 aliases)");
}
