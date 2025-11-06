// plugins/core_plugin.cpp
#include "plugin.hpp"
#include "logger.hpp"
#include "command_registry.hpp"  // ✅ NEW: Command registry

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
#include "commands_tasks.hpp"
#include "commands_question.hpp"

// Helper macro for registering with metadata
#define REGISTER_TOOL(cmd, handler, desc, cat, isInfo) \
    grim_register_command(cmd, handler); \
    GRIM::CommandRegistry::registerTool(cmd, { \
        cmd, desc, cmd, cat, isInfo, !isInfo, {}, {}, {} \
    });

// One shared registration list used by both host and plugin builds.
static void register_all_commands() {
    LOG_DEBUG("Plugin", "Registering core commands...");
    
    // Core AI + NLP
    REGISTER_TOOL("reload_nlp", cmdReloadNlp, 
                  "Reload NLP grammar rules from disk", 
                  "system", true);
    REGISTER_TOOL("clear_context", cmdClearContext, 
                  "Clear conversation context and history", 
                  "system", false);
    REGISTER_TOOL("grim_ai", cmdGrimAi, 
                  "Send a direct query to the AI backend", 
                  "information", true);
    
    // Application control
    LOG_DEBUG("Plugin", "Registering 'open' command");
    REGISTER_TOOL("open", cmdOpenApp,
                  "Open an application or file",
                  "action", false);
    LOG_DEBUG("Plugin", "Registering 'search' command");
    REGISTER_TOOL("search", cmdSearchWeb,
                  "Search the web for information",
                  "information", true);

    // Filesystem / terminal-like
    REGISTER_TOOL("pwd", cmdShowPwd,
                  "Show current working directory",
                  "information", true);
    REGISTER_TOOL("cd", cmdChangeDir,
                  "Change current directory",
                  "action", false);
    REGISTER_TOOL("ls", cmdListDir,
                  "List directory contents",
                  "information", true);
    REGISTER_TOOL("mkdir", cmdMakeDir,
                  "Create a new directory",
                  "action", false);
    REGISTER_TOOL("rm", cmdRemoveFile,
                  "Remove a file or directory",
                  "action", false);
    REGISTER_TOOL("clean", cmdClean,
                  "Clean console history",
                  "system", false);
    REGISTER_TOOL("help", cmdShowHelp,
                  "Show available commands",
                  "information", true);

    // Aliases
    REGISTER_TOOL("alias list", cmdAliasList,
                  "List all defined command aliases",
                  "information", true);
    REGISTER_TOOL("alias info", cmdAliasInfo,
                  "Show information about a specific alias",
                  "information", true);
    REGISTER_TOOL("alias refresh", cmdAliasRefresh,
                  "Refresh and reload aliases from disk",
                  "system", false);

    // Memory commands
    LOG_DEBUG("Plugin", "Registering memory commands");
    REGISTER_TOOL("remember", cmdRemember,
                  "Store information in long-term memory",
                  "action", false);
    REGISTER_TOOL("recall", cmdRecall,
                  "Retrieve information from memory",
                  "information", true);
    REGISTER_TOOL("forget", cmdForget,
                  "Remove information from memory",
                  "action", false);

    // UI Controls
    LOG_DEBUG("Plugin", "Registering UI commands");
    REGISTER_TOOL("console", cmdToggleOverlayConsole,
                  "Toggle overlay console visibility",
                  "ui", false);
    REGISTER_TOOL("toggle_settings", cmdToggleSettings,
                  "Toggle settings panel visibility",
                  "ui", false);

    // System information
    REGISTER_TOOL("system_info", cmdSystemInfo,
                  "Display system hardware and software information",
                  "information", true);

    // Test commands (defined in commands_system.cpp)
    LOG_DEBUG("Plugin", "Registering 'test_intent' command");
    REGISTER_TOOL("test_intent", cmdTestIntent,
                  "Test intent classification on input",
                  "system", true);
    LOG_DEBUG("Plugin", "Registering 'test_nlp' command");
    REGISTER_TOOL("test_nlp", cmdTestNlp,
                  "Test NLP grammar parsing",
                  "system", true);

    // Voice commands
    LOG_DEBUG("Plugin", "Registering voice commands");
    REGISTER_TOOL("test_tts", cmd_testTTS,
                  "Test text-to-speech output",
                  "system", false);
    REGISTER_TOOL("list_voices", cmd_listVoices,
                  "List available TTS voices",
                  "information", true);
    REGISTER_TOOL("nevermind", cmdNevermind,
                  "Cancel the last command or action",
                  "action", false);
    
    // ? NEW: Speaker embedding commands (XTTS v2)
    LOG_DEBUG("Plugin", "Registering speaker embedding commands");
    REGISTER_TOOL("create_embedding", cmd_createEmbedding,
                  "Create voice embedding for TTS",
                  "system", false);
    REGISTER_TOOL("list_embeddings", cmd_listEmbeddings,
                  "List available voice embeddings",
                  "information", true);

    // NLP management commands
    LOG_DEBUG("Plugin", "Registering NLP management commands");
    REGISTER_TOOL("nlp_stats", cmdNlpStats,
                  "Show NLP grammar statistics",
                  "information", true);
    REGISTER_TOOL("nlp_learn", cmdNlpLearn,
                  "Teach NLP a new pattern",
                  "system", false);
    REGISTER_TOOL("nlp_save", cmdNlpSave,
                  "Save learned NLP patterns",
                  "system", false);
    
    // ✅ NEW: Command registry management
    LOG_DEBUG("Plugin", "Registering command registry commands");
    REGISTER_TOOL("list_tools", cmdListTools,
                  "List all registered commands/tools",
                  "information", true);
    REGISTER_TOOL("tool_info", cmdToolInfo,
                  "Show detailed information about a tool",
                  "information", true);
    REGISTER_TOOL("tool_stats", cmdToolStats,
                  "Show tool usage statistics",
                  "information", true);

    // Settings info command (renamed to avoid conflict)
    LOG_DEBUG("Plugin", "Registering settings info command");
    REGISTER_TOOL("settings", cmdSettings,
                  "Show current settings and configuration",
                  "information", true);

    // ? NEW: GRIM system management commands
    LOG_DEBUG("Plugin", "Registering GRIM system commands");
    REGISTER_TOOL("clear_tts_cache", cmdClearCache,
                  "Clear TTS audio cache",
                  "system", false);
    REGISTER_TOOL("reset_tts_cache", cmdResetCache,
                  "Reset TTS cache and regenerate common phrases",
                  "system", false);

    // ✅ NEW: Perception/Vision commands
    LOG_DEBUG("Plugin", "Registering perception commands");
    REGISTER_TOOL("perception_what_see", cmdPerceptionWhat,
                  "Describe what's visible on screen",
                  "information", true);
    REGISTER_TOOL("perception_read_text", cmdPerceptionRead,
                  "Read text from screen using OCR",
                  "information", true);
    REGISTER_TOOL("perception_detect_objects", cmdPerceptionDetect,
                  "Detect and locate objects on screen",
                  "information", true);

    // ✅ NEW: Input Control commands
    LOG_DEBUG("Plugin", "Registering input control commands");
    REGISTER_TOOL("move_mouse", cmdInputMoveMouse,
                  "Move mouse cursor to position",
                  "action", false);
    REGISTER_TOOL("click", cmdInputClick,
                  "Perform mouse click",
                  "action", false);
    REGISTER_TOOL("type", cmdInputType,
                  "Type text at current position",
                  "action", false);
    REGISTER_TOOL("key", cmdInputKey,
                  "Press keyboard key or combination",
                  "action", false);
    REGISTER_TOOL("click_on", cmdInputClickOn,
                  "Click on detected screen element",
                  "action", false);

    // ✅ NEW: Task management commands
    LOG_DEBUG("Plugin", "Registering task management commands");
    REGISTER_TOOL("execute_task", cmdExecuteTask,
                  "Execute a task",
                  "action", false);
    REGISTER_TOOL("task_status", cmdTaskStatus,
                  "Check task execution status",
                  "information", true);
    REGISTER_TOOL("task_cancel", cmdTaskCancel,
                  "Cancel running task",
                  "action", false);

    // ✅ NEW: Timer commands
    LOG_DEBUG("Plugin", "Registering timer commands");
    REGISTER_TOOL("set_timer", cmdSetTimer,
                  "Set a countdown timer",
                  "action", false);

    // ✅ NEW: Question handler
    LOG_DEBUG("Plugin", "Registering question handler");
    REGISTER_TOOL("question", cmdQuestion,
                  "Ask GRIM a question (searches memory and external sources)",
                  "information", true);

    // ✅ NEW: Additional voice commands (advanced TTS)
    LOG_DEBUG("Plugin", "Registering additional voice commands");
    REGISTER_TOOL("voice", cmdVoice,
                  "General voice command handler",
                  "system", false);
    REGISTER_TOOL("voice_stream", cmdVoiceStream,
                  "Stream voice output",
                  "system", false);
    REGISTER_TOOL("test_sapi", cmd_testSAPI,
                  "Test SAPI TTS backend",
                  "system", false);
    REGISTER_TOOL("tts_device", cmd_ttsDevice,
                  "Configure TTS output device",
                  "system", false);

    // ✅ NEW: AI backend management
    LOG_DEBUG("Plugin", "Registering AI backend command");
    REGISTER_TOOL("ai_backend", cmdAiBackend,
                  "Show or change active AI backend",
                  "system", false);

    LOG_DEBUG("Plugin", "Core command registration complete");
    LOG_DEBUG("CommandRegistry", "Total registered tools: " + 
              std::to_string(GRIM::CommandRegistry::getToolCount()));
    
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


