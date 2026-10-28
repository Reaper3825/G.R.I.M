#include "commands_core.hpp"
#include "MMO/Core/ToolRegistry.hpp"

// Only explicitly selected registered tools execute here. Intent and policy
// selection remain the caller's responsibility.
CommandResult dispatchCommand(const std::string& cmd, const std::string& arg) {
    ensureCorePluginsRegistered();
    return GRIM::MMO::ToolRegistry::instance().invoke(cmd, arg);
}
