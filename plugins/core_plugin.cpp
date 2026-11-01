// plugins/core_plugin.cpp
#include "logger.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <stdexcept>

#include "commands_ui.hpp"
#include "commands_osint.hpp"

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
    tools.back().descriptor.usage = "ui.create_surface {t} <kind> <title>";
    tools.back().descriptor.bindings = {{"t", "t"}};
    REGISTER_TOOL("ui.destroy_surface", cmdDestroySurface,
                  "Permanently remove a UI surface",
                  "ui", false);
    tools.back().descriptor.usage = "ui.destroy_surface {t}";
    tools.back().descriptor.bindings = {{"t", "t"}};

    // System information
    REGISTER_TOOL("ui.show_osint_results", cmdOsintShowUI,
                  "Show the host-owned OSINT findings panel for a username", "ui", false);
    tools.back().descriptor.usage_type = GRIM::MMO::ToolUsage::Ignore;
    tools.back().descriptor.usage = "ui.show_osint_results {t}";
    tools.back().descriptor.bindings = {{"t", "t"}};

    // System information is supplied by system_info_plugin.

    // Voice and tool inspection commands are supplied by native plugins.

    // Observation and input tools are supplied by perception_plugin and input_plugin.

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
