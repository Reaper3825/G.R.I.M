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

extern "C" GRIM_PLUGIN_API void registerGrimPlugin() {
    LOG_DEBUG("Plugin", "OSINT Self-Audit plugin v1.0.0 loading");

    grim_register_command("profile_self", wrapProfileSelf);
    grim_register_command("sherlock_sweep", wrapSherlockSweep);
    grim_register_command("osint_report", wrapOsintReport);

    LOG_DEBUG("Plugin", "OSINT commands registered");
}
