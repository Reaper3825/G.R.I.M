// plugins/core_plugin.cpp
#include "plugin.hpp"

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
    // Core AI + NLP
    grim_register_command("reload_nlp",   cmdReloadNlp);
    grim_register_command("grim_ai",      cmdGrimAi);

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

    // (Add any additional core commands here; keep both paths in sync.)
}

// --- Plugin DLL entry (only when building the DLL) ---
#if defined(GRIM_BUILD_PLUGIN)
extern "C" GRIM_PLUGIN_API void registerGrimPlugin() {
    register_all_commands();
}
#endif

// --- Built into host once (when compiling into GRIM exe) ---
#if defined(GRIM_BUILD_HOST)
void registerCorePlugin() {
    register_all_commands();
}
#endif
