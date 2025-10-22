// plugins/core_plugin.cpp
#include "plugin.hpp"
#include "logger.hpp"

#include "commands_interface.hpp"
#include "commands_filesystem.hpp"
#include "commands_aliases.hpp"
#include "commands_memory.hpp"
#include "commands_ai.hpp"
#include "commands_system.hpp"
#include "commands_timers.hpp"
#include "commands_voice.hpp"

// One shared registration list used by both host and plugin builds.
static void register_all_commands() {
    LOG_DEBUG("Plugin", "Registering core commands...");
    
    // Core AI + NLP
    grim_register_command("reload_nlp",   cmdReloadNlp);
    grim_register_command("grim_ai",      cmdGrimAi);
    
    // Application control
    LOG_DEBUG("Plugin", "Registering 'open' command");
    grim_register_command("open",         cmdOpenApp);
    LOG_DEBUG("Plugin", "Registering 'search' command");
    grim_register_command("search",       cmdSearchWeb);

    // Filesystem / terminal-like
    grim_register_command("pwd",          cmdShowPwd);
    grim_register_command("cd",           cmdChangeDir);
    grim_register_command("ls",           cmdListDir);
    grim_register_command("mkdir",        cmdMakeDir);
    grim_register_command("rm",           cmdRemoveFile);
    grim_register_command("clean",        cmdClean);
    grim_register_command("help",         cmdShowHelp);

    // Aliases
    grim_register_command("alias list",    cmdAliasList);
    grim_register_command("alias info",    cmdAliasInfo);
    grim_register_command("alias refresh", cmdAliasRefresh);

    // ? Test commands (defined in commands_system.cpp)
    LOG_DEBUG("Plugin", "Registering 'test_intent' command");
    grim_register_command("test_intent",   cmdTestIntent);
    LOG_DEBUG("Plugin", "Registering 'test_nlp' command");
    grim_register_command("test_nlp",      cmdTestNlp);

    // ? NLP management commands
    LOG_DEBUG("Plugin", "Registering NLP management commands");
    grim_register_command("nlp_stats",     cmdNlpStats);
    grim_register_command("nlp_learn",     cmdNlpLearn);
    grim_register_command("nlp_save",      cmdNlpSave);

    // ? Settings command (defined in commands_system.cpp)
    LOG_DEBUG("Plugin", "Registering settings command");
    grim_register_command("settings",      cmdSettings);

    // ? OSINT self-audit commands (defensive only) - DISABLED until CMake picks up file
    // LOG_DEBUG("Plugin", "Registering OSINT commands");
    // extern CommandResult cmdProfileSelf(const std::string&);
    // extern CommandResult cmdSherlockSweep(const std::string&);
    // extern CommandResult cmdOsintReport(const std::string&);
    // grim_register_command("profile_self",    cmdProfileSelf);
    // grim_register_command("sherlock_sweep",  cmdSherlockSweep);
    // grim_register_command("osint_report",    cmdOsintReport);

    LOG_DEBUG("Plugin", "Core command registration complete");
    
    // (Add any additional core commands here; keep both paths in sync.)
}

// --- Plugin DLL entry (only when building the DLL) ---
#if defined(GRIM_BUILD_PLUGIN)
extern "C" GRIM_PLUGIN_API void registerGrimPlugin() {
    LOG_DEBUG("Plugin", "registerGrimPlugin() called (DLL mode)");
    register_all_commands();
}
#endif

// --- Built into host once (when compiling into GRIM exe) ---
#if defined(GRIM_BUILD_HOST)
void registerCorePlugin() {
    // IMMEDIATE log to verify this code path exists
    std::cerr << "[CRITICAL] registerCorePlugin() ENTRY POINT HIT!" << std::endl;
    LOG_DEBUG("Plugin", "registerCorePlugin() called (HOST mode)");
    register_all_commands();
}
#endif
