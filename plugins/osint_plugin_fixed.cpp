// plugins/osint_plugin.cpp - OSINT Self-Audit Plugin
#include "pch.hpp"
#include "commands/commands_osint.hpp"
#include "core/plugin.hpp"
#include "logger.hpp"

extern "C" {

// Plugin metadata
 {
    return "OSINT Self-Audit";
}

 {
    return "1.0.0";
}

 {
    return "Defensive OSINT tools for privacy self-audits. ?? Your accounts only!";
}

// Plugin initialization
 {
    LOG_DEBUG("Plugin", "Initializing OSINT Self-Audit plugin");
    
    // Register OSINT commands
    grim_register_command("profile_self",    cmdProfileSelf);
    grim_register_command("sherlock_sweep",  cmdSherlockSweep);
    grim_register_command("osint_report",    cmdOsintReport);
    
    LOG_DEBUG("Plugin", "OSINT commands registered: profile_self, sherlock_sweep, osint_report");
}

// Plugin cleanup
 {
    LOG_DEBUG("Plugin", "Shutting down OSINT plugin");
}

} // extern "C"
