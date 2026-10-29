// plugins/core_plugin.cpp
#include "logger.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <stdexcept>

#include "commands_ai.hpp"
#include "commands_system.hpp"
#include "commands_voice.hpp"
#include "commands_ui.hpp"
#include "commands_osint.hpp"
#include "commands_grim.hpp"
#include "commands_perception.hpp"
#include "commands_question.hpp"

// Built-ins publish metadata and execution together in the sole tool registry.
static GRIM::MMO::ToolRegistration builtinTool(const char* id, GRIM::MMO::ToolHandler handler,
                                const char* description, const char* category, bool informational) {
    GRIM::MMO::ToolDescriptor tool;
    tool.tool_id = tool.display_name = tool.usage = id;
    tool.description = description;
    tool.category = category;
    tool.provider_name = "builtin";
    tool.version = "1.0";
    tool.is_informational = informational;
    tool.needs_confirmation = !informational;
    return {std::move(tool), std::move(handler)};
}
#define REGISTER_TOOL(cmd, handler, desc, cat, isInfo) \
    tools.push_back(builtinTool(cmd, handler, desc, cat, isInfo))

// Built into the host; dynamic plugins use core/plugin_api.hpp.
static void register_all_commands() {
    std::vector<GRIM::MMO::ToolRegistration> tools;
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
    
    // Application/browser commands are supplied by application_plugin.

    // Filesystem and utility tools are supplied by plugins discovered from disk.

    // Alias commands are supplied by aliases_plugin; the host owns alias state.

    // Manual memory commands are supplied by memory_plugin.

    // UI Controls
    LOG_DEBUG("Plugin", "Registering UI commands");
    REGISTER_TOOL("console", cmdToggleOverlayConsole,
                  "Toggle overlay console visibility",
                  "ui", false);
    REGISTER_TOOL("toggle_settings", cmdToggleSettings,
                  "Toggle settings panel visibility",
                  "ui", false);

    // MMO UI surface tools
    REGISTER_TOOL("ui.create_surface", cmdCreateSurface,
                  "Create a new UI surface (panel, popup, modal, toast, tool window, inspector)",
                  "ui", false);
    REGISTER_TOOL("ui.show_surface", cmdShowSurface,
                  "Show a previously created UI surface",
                  "ui", false);
    REGISTER_TOOL("ui.hide_surface", cmdHideSurface,
                  "Hide a visible UI surface without destroying it",
                  "ui", false);
    REGISTER_TOOL("ui.destroy_surface", cmdDestroySurface,
                  "Permanently remove a UI surface",
                  "ui", false);

    // System information
    REGISTER_TOOL("ui.show_osint_results", cmdOsintShowUI,
                  "Show the host-owned OSINT findings panel for a username", "ui", false);

    // System information is supplied by system_info_plugin.

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
    
    // Tool inspection is supplied by tool_inspection_plugin.

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

    // Observation commands are supplied by perception_plugin; input stays here.

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

    // User timer commands are supplied by timers_plugin.

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

    if (!GRIM::MMO::ToolRegistry::instance().registerBatch({}, tools))
        throw std::runtime_error("Conflicting built-in tool registration");
    LOG_DEBUG("Plugin", "Core command registration complete");
    LOG_DEBUG("ToolRegistry", "Total registered tools: " +
              std::to_string(GRIM::MMO::ToolRegistry::instance().toolCount()));
    
    // Add additional built-in commands here.
}

// --- Built into host once (when compiling into GRIM exe) ---
#if defined(GRIM_BUILD_HOST)
void registerCorePlugin() {
    // IMMEDIATE log to verify this code path exists
    std::cerr << "[CRITICAL] registerCorePlugin() ENTRY POINT HIT!" << std::endl;
    LOG_DEBUG("Plugin", "registerCorePlugin() called (HOST mode)");
    register_all_commands();
}
#endif


