#include "core/plugin_manager.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/commands_core.hpp"
#include <filesystem>
#include <stdexcept>
void runInputPluginTests(const std::filesystem::path& directory) {
 if(directory.empty()) return;
 auto check=[](bool ok,const char* message){ if(!ok) throw std::runtime_error(message); };
 namespace fs=std::filesystem;
 const auto folder=fs::absolute(directory).parent_path()/"input-plugin-test-deployment";
 fs::create_directories(folder);
 fs::copy_file(fs::absolute(directory)/"input_plugin.dll",folder/"input_plugin.dll",fs::copy_options::overwrite_existing);
 auto& registry=GRIM::MMO::ToolRegistry::instance(); auto baseline=registry.toolCount();
 PluginManager::initialize(folder.string());
 check(registry.toolCount()==baseline+4,"Input DLL registration failed");
 for(const auto* id:{"move_mouse","click","type","key"}) {
  auto tool=registry.getTool(id);
  check(tool && tool->provider.name=="Input Plugin" && tool->permission_bits==GRIM_PERM_INPUT,"Input ownership/permissions changed");
 }
 // Invalid requests never inject native input. Device effects are tested with a provider-local fixture.
 check(!dispatchCommand("type","").success,"Empty typing accepted");
 check(!dispatchCommand("move_mouse","bad").success,"Malformed move accepted");
 PluginManager::unloadAll(); check(registry.toolCount()==baseline,"Input tools survived unload");
 PluginManager::initialize(folder.string());
 fs::remove(folder/"input_plugin.dll"); PluginManager::checkForHotReload();
 check(registry.toolCount()==baseline,"Input tools survived removal"); PluginManager::unloadAll();
}
