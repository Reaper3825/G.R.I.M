#include "plugin.hpp"
#include "commands/command_result.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "logger.hpp"

extern "C" GRIM_HOST_API
void grim_register_command(const char* name, PluginCommandFunc func) {
    if (!name || !*name || !func) {
        LOG_ERROR("PluginSystem", "Invalid host command registration");
        return;
    }
    GRIM::MMO::ToolDescriptor tool;
    tool.tool_id = tool.display_name = tool.usage = name;
    tool.provider_name = "builtin";
    tool.description = std::string("Host command: ") + name;
    tool.category = "system";
    if (!GRIM::MMO::ToolRegistry::instance().registerTool(tool, func))
        LOG_ERROR("PluginSystem", std::string("Duplicate host command: ") + name);
}
extern "C" GRIM_HOST_API
void grim_unregister_command(const char* name) {
    if (name) GRIM::MMO::ToolRegistry::instance().unregisterTool(
        name, "builtin", GRIM::MMO::ToolProviderType::Builtin);
}
