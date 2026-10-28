// Host-only loader lifecycle test with tiny fixture DLLs and API service sinks.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "commands/command_registry.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
std::set<std::string> active;
std::vector<std::string> events;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void fixtureLog(GrimLogLevel, const char* name) {
    check(active.count(name) == 1, "Plugin shutdown ran after its context was removed");
    events.push_back(std::string("shutdown:") + name);
}
}
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string& message) { throw std::runtime_error(message); }
namespace PluginAPI {
GrimPluginAPI* createPluginAPI(const std::string& name, const std::string&, uint32_t) {
    check(active.insert(name).second, "Plugin loaded more than once");
    static GrimPluginAPI api{};
    api.api_version = GRIM_API_VERSION;
    api.log = fixtureLog;
    return &api;
}
void cleanupPluginContext(const std::string& name) {
    check(active.erase(name) == 1, "Cleanup used DLL filename instead of declared plugin identity");
    check(std::find(events.begin(), events.end(), "shutdown:" + name) != events.end(), "Cleanup ran before shutdown");
    events.push_back("cleanup:" + name);
}
}
int main(int argc, char** argv) try {
    check(argc == 2, "Expected fixture directory");
    PluginManager::initialize(argv[1]);
    check(active.size() == 2, "Both fixture plugins must load");
    PluginManager::unloadAll();
    check(active.empty() && events.size() == 4, "Shutdown did not clean up every plugin");
    PluginManager::unloadAll();
    check(events.size() == 4, "Repeated shutdown called unloaded plugins");
    PluginManager::initialize(argv[1]);
    check(active.size() == 2, "Shutdown retained stale loaded-plugin entries");
    PluginManager::unloadAll();
    check(active.empty() && events.size() == 8, "Reloaded fixtures did not shut down");

    // The retained startup staging API still registers, replaces, and sorts.
    GRIM::CommandRegistry::ToolMetadata tool;
    tool.description = "original"; tool.category = "z";
    GRIM::CommandRegistry::registerTool("test_tool", tool);
    tool.description = "replacement"; tool.category = "a";
    GRIM::CommandRegistry::registerTool("test_tool", tool);
    GRIM::CommandRegistry::registerTool("other_tool", tool);
    const auto tools = GRIM::CommandRegistry::getAllTools();
    check(GRIM::CommandRegistry::getToolCount() == 2 && tools.size() == 2, "Staging registry count changed");
    check(tools[0].name == "other_tool" && tools[1].description == "replacement", "Staging replacement/sorting changed");
    std::cout << "Plugin shutdown and retained registry tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
