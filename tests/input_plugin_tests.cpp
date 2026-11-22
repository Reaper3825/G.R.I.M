#include "core/plugin_manager.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <filesystem>
#include <stdexcept>
void runInputPluginTests(const std::filesystem::path& directory) {
 if(directory.empty()) return;
 auto check=[](bool ok,const char* message){ if(!ok) throw std::runtime_error(message); };
 namespace fs=std::filesystem;
 const auto folder=fs::absolute(directory).parent_path()/"input-plugin-test-deployment";
 fs::create_directories(folder / "input_plugin");
 fs::copy_file(fs::absolute(directory)/"input_plugin.dll",folder/"input_plugin/input_plugin.dll",fs::copy_options::overwrite_existing);
 fs::copy_file(fs::absolute(directory)/"input_plugin.tools.json",folder/"input_plugin/input_plugin.tools.json",fs::copy_options::overwrite_existing);
 auto& registry=GRIM::MMO::ToolRegistry::instance(); auto baseline=registry.toolCount();
 PluginManager::loadBatch(folder.string());
 check(registry.toolCount()==baseline+4,"Input DLL registration failed");
 for(const auto* id:{"move_mouse","click","type","key"}) {
  auto tool=registry.getTool(id);
  check(tool && tool->provider.name=="Input Plugin","Input ownership changed");
 }
 // Invalid requests never inject native input. Device effects are tested with a provider-local fixture.
 check(registry.invoke("type","").at("status") == "failed","Empty typing accepted");
 check(registry.invoke("move_mouse","bad").at("status") == "failed","Malformed move accepted");
 PluginManager::shutdownBatch(); check(registry.toolCount()==baseline,"Input tools survived unload");
 PluginManager::loadBatch(folder.string());
 PluginManager::shutdownBatch();
 fs::remove(folder/"input_plugin/input_plugin.dll");
 fs::remove(folder/"input_plugin/input_plugin.tools.json");
 fs::remove(folder/"input_plugin");
 check(registry.toolCount()==baseline,"Input tools survived removal"); PluginManager::shutdownBatch();
}
