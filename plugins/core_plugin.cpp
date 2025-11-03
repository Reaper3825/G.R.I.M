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
#include "commands_ui.hpp"
#include "commands_grim.hpp"  // ? NEW: GRIM system commands
#include "commands_perception.hpp"

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

    // Memory commands
    LOG_DEBUG("Plugin", "Registering memory commands");
    grim_register_command("remember",      cmdRemember);
    grim_register_command("recall",        cmdRecall);
    grim_register_command("forget",        cmdForget);

    // UI Controls
    LOG_DEBUG("Plugin", "Registering UI commands");
    grim_register_command("console",       cmdToggleOverlayConsole);
    grim_register_command("ui_console",    cmdToggleOverlayConsole);
    grim_register_command("toggle_console", cmdToggleOverlayConsole);
    grim_register_command("settings",      cmdToggleSettings);
    grim_register_command("settings_ui",   cmdToggleSettings);
    grim_register_command("toggle_settings", cmdToggleSettings);

    // Test commands (defined in commands_system.cpp)
    LOG_DEBUG("Plugin", "Registering 'test_intent' command");
    grim_register_command("test_intent",   cmdTestIntent);
    LOG_DEBUG("Plugin", "Registering 'test_nlp' command");
    grim_register_command("test_nlp",      cmdTestNlp);

    // Voice commands
    LOG_DEBUG("Plugin", "Registering voice commands");
    grim_register_command("test_tts",      cmd_testTTS);
    grim_register_command("list_voices",   cmd_listVoices);
    grim_register_command("nevermind",     cmdNevermind);
    
    // ? NEW: Speaker embedding commands (XTTS v2)
    LOG_DEBUG("Plugin", "Registering speaker embedding commands");
    grim_register_command("create_embedding", cmd_createEmbedding);
    grim_register_command("list_embeddings",  cmd_listEmbeddings);

    // NLP management commands
    LOG_DEBUG("Plugin", "Registering NLP management commands");
    grim_register_command("nlp_stats",     cmdNlpStats);
    grim_register_command("nlp_learn",     cmdNlpLearn);
    grim_register_command("nlp_save",      cmdNlpSave);

    // Settings info command (renamed to avoid conflict)
    LOG_DEBUG("Plugin", "Registering settings info command");
    grim_register_command("settings_info", cmdSettings);  // ? Renamed from "settings"
    grim_register_command("config_info",   cmdSettings);  // ? Alternative name

    // ? NEW: GRIM system management commands
    LOG_DEBUG("Plugin", "Registering GRIM system commands");
    grim_register_command("clear_cache",   cmdClearCache);
    grim_register_command("reset_cache",   cmdResetCache);

    // ✅ NEW: Perception/Vision commands
    LOG_DEBUG("Plugin", "Registering perception commands");
    grim_register_command("perception_what_see",      cmdPerceptionWhat);
    grim_register_command("perception_read_text",     cmdPerceptionRead);
 grim_register_command("perception_detect_objects", cmdPerceptionDetect);

    // ✅ NEW: Input Control commands
    LOG_DEBUG("Plugin", "Registering input control commands");
    grim_register_command("move_mouse",    cmdInputMoveMouse);
    grim_register_command("click",         cmdInputClick);
    grim_register_command("type",          cmdInputType);
    grim_register_command("key",           cmdInputKey);
    grim_register_command("click_on",      cmdInputClickOn);

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


