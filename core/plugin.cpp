#include "plugin.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"
#include <unordered_map>
#include <mutex>
#include <string>

// Make this a true definition with external linkage (NO 'static')
#if defined(GRIM_BUILD_PLUGIN)
extern std::unordered_map<std::string, CommandFunc> commandMap; // plugin just references
#else
std::unordered_map<std::string, CommandFunc> commandMap;        // host owns it
#endif

static std::mutex regMutex;

// ---------------- Host exports (stable ABI) ----------------
extern "C" GRIM_HOST_API
void grim_register_command(const char* name, PluginCommandFunc func) {
    std::lock_guard<std::mutex> lock(regMutex);
    commandMap[std::string(name)] = func;
    LOG_DEBUG("PluginSystem", std::string("Registered command: ") + name);
}

extern "C" GRIM_HOST_API
void grim_unregister_command(const char* name) {
    std::lock_guard<std::mutex> lock(regMutex);
    commandMap.erase(std::string(name));
    LOG_DEBUG("PluginSystem", std::string("Unregistered command: ") + name);
}

// ---------------- Legacy host-only helpers ----------------
#if defined(GRIM_BUILD_HOST)
void registerPluginCommands(const std::vector<PluginCommand>& cmds) {
    std::lock_guard<std::mutex> lock(regMutex);
    for (const auto& c : cmds) {
        commandMap[c.name] = c.func;
        LOG_DEBUG("PluginSystem", "Registered command: " + c.name);
    }
}

void unregisterPluginCommands(const std::string& pluginName) {
    (void)pluginName; // reserved
}
#endif

// Optional accessor if your dispatcher needs it
const std::unordered_map<std::string, CommandFunc>& grim_getCommandMap() {
    return commandMap;
}
