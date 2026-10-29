// plugins/core_plugin.cpp
#include "logger.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <stdexcept>

#include "commands_ui.hpp"
#include "commands_osint.hpp"
#include "commands_perception.hpp"

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
    
    // Retired AI/NLP/admin handlers remain in commands/ for host use.
    // See docs/RETIRED_AGENT_TOOLS.md; they are not registered as agent tools.

    // Application/browser commands are supplied by application_plugin.

    // Filesystem and utility tools are supplied by plugins discovered from disk.

    // Alias commands are supplied by aliases_plugin; the host owns alias state.

    // Manual memory commands are supplied by memory_plugin.

    // Generic panel operations are supplied by ui_plugin.
    // The old Console/Settings toggles and surface visibility wrappers are retired.

    // MMO UI surface tools
    REGISTER_TOOL("ui.create_surface", cmdCreateSurface,
                  "Create a new UI surface (panel, popup, modal, toast, tool window, inspector)",
                  "ui", false);
    REGISTER_TOOL("ui.destroy_surface", cmdDestroySurface,
                  "Permanently remove a UI surface",
                  "ui", false);

    // System information
    REGISTER_TOOL("ui.show_osint_results", cmdOsintShowUI,
                  "Show the host-owned OSINT findings panel for a username", "ui", false);

    // System information is supplied by system_info_plugin.

    // Voice and tool inspection commands are supplied by native plugins.

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
