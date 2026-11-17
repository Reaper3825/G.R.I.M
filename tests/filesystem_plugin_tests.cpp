// Real DLL -> PluginManager -> API -> ToolRegistry -> filesystem, in a private sandbox.
#include "core/plugin_manager.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "MMO/Core/Descriptors/RegisteredResource.hpp"
#include "commands/command_result.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void runFilesystemPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto processDirectory = fs::current_path();
    const auto directoryResource = [&] {
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if (group.provider.name == "Filesystem Plugin")
                for (const auto& resource : group.resources)
                    if (resource.resource_id == "working_directory")
                        return GRIM::MMO::RegisteredResource{resource, group.provider};
        return GRIM::MMO::RegisteredResource{};
    };
    const auto utf8 = [](const fs::path& path) { const auto text = path.u8string(); return std::string(text.begin(), text.end()); };

    const auto sandbox = fs::temp_directory_path() / ("grim-filesystem-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(sandbox), "Cannot create private filesystem test directory");
    struct Cleanup {
        fs::path path;
        fs::path originalDirectory;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            fs::current_path(originalDirectory, error);
            // Only known fixture paths are removed. No recursive cleanup.
            for (const auto* name : {"plugins/filesystem_plugin.dll", "plugins/filesystem_plugin.tools.json", "sample.txt", "new folder", "nonempty/keep.txt", "nonempty", "plugins"})
                fs::remove(path / name, error);
            fs::remove(path, error);
        }
    } cleanup{sandbox, processDirectory};
    fs::current_path(sandbox);
    const auto baseDirectory = fs::current_path();
    fs::create_directory(sandbox / "plugins");
    const auto deployed = sandbox / "plugins/filesystem_plugin.dll";
    fs::copy_file(pluginDirectory / "filesystem_plugin.dll", deployed);
    fs::copy_file(pluginDirectory / "filesystem_plugin.tools.json", deployed.parent_path() / "filesystem_plugin.tools.json");
    PluginManager::initialize((sandbox / "plugins").string());
    check(registry.toolCount() == baseline + 3, "Filesystem DLL did not register exactly three tools");
    check(!registry.isRegistered("cd") && !registry.isRegistered("pwd"), "Retired navigation tools registered");
    for (const auto* id : {"ls", "mkdir", "rm"}) {
        const auto tool = registry.getTool(id);
        check(tool && tool->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              tool->provider.name == "Filesystem Plugin" && fs::path(tool->provider.path) == deployed,
              "Filesystem tool has incorrect provider ownership");
        check(!tool->usage.empty() && tool->permission_bits == GRIM_PERM_FILESYSTEM,
              "Filesystem tool metadata incomplete");
        const bool informational = std::string(id) == "ls";
        check(tool->is_informational == informational && tool->needs_confirmation != informational &&
              tool->parameters.size() == (informational ? 0 : 1), "Filesystem parameter/policy metadata incorrect");
    }
    const auto invoke = [&](const char* id, const std::string& args = "") { return registry.invoke(id, args); };
    const auto initialResource = directoryResource();
    check(initialResource.kind == "filesystem.directory" && initialResource.runtime_id == 1 &&
          initialResource.state.at("path") == utf8(baseDirectory), "Initial working directory resource missing");
    const auto initialRevision = GRIM::MMO::ProviderRegistry::instance().resourceVersion();
    check(invoke("mkdir", "\"new folder\"").success && fs::is_directory(sandbox / "new folder"), "Quoted mkdir failed");
    check(!invoke("mkdir", "new folder").success, "Duplicate mkdir succeeded");
    check(!invoke("mkdir", "missing/child").success, "mkdir created missing parents");
    { std::ofstream file(sandbox / "sample.txt"); file << "fixture"; }
    check(invoke("ls").message.find("sample.txt") != std::string::npos, "ls omitted file");
    for (const auto* id : {"mkdir", "rm"})
        check(invoke(id).errorCode == "ERR_FS_NO_ARGUMENT", "Missing argument not rejected");
    check(invoke("rm", std::string("bad\0path", 8)).errorCode == "ERR_FS_INVALID_PATH", "NUL path accepted");
    fs::create_directory(sandbox / "nonempty");
    { std::ofstream file(sandbox / "nonempty/keep.txt"); file << "keep"; }
    check(!invoke("rm", "nonempty").success && fs::exists(sandbox / "nonempty/keep.txt"), "rm removed nonempty directory");
    check(invoke("rm", "sample.txt").success && !fs::exists(sandbox / "sample.txt"), "rm file failed");
    check(invoke("rm", "new folder").success && !fs::exists(sandbox / "new folder"), "rm empty directory failed");
    check(invoke("rm", "missing").errorCode == "ERR_FS_NOT_FOUND", "Missing rm path not reported");
    check(fs::current_path() == baseDirectory &&
          GRIM::MMO::ProviderRegistry::instance().resourceVersion() == initialRevision,
          "Filesystem tools changed the working directory or published resources");
    // A native load failure after cutover leaves the provider absent.
    { std::ofstream invalid(deployed, std::ios::binary | std::ios::trunc); invalid << "invalid native library"; }
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && directoryResource().resource_id.empty() &&
          invoke("ls").errorCode == "ERR_UNKNOWN_CMD",
          "Failed native load retained filesystem tools or resources");
    // A fresh DLL instance publishes its own directory.
    fs::copy_file(pluginDirectory / "filesystem_plugin.dll", deployed, fs::copy_options::overwrite_existing);
    fs::last_write_time(deployed, fs::last_write_time(deployed) + std::chrono::seconds(2));
    PluginManager::checkForHotReload();
    check(directoryResource().provider.instance != initialResource.provider.instance &&
          directoryResource().state.at("path") == utf8(baseDirectory) &&
          invoke("ls").success, "Reload retained old generation or lost the replacement resource");
    fs::remove(deployed);
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && directoryResource().resource_id.empty(), "Removing filesystem DLL retained tools/resources");
    check(invoke("ls").errorCode == "ERR_UNKNOWN_CMD", "Removed filesystem tool still executable");
    std::cout << "Filesystem plugin operations, directory resources, failed replacement, reload and removal tests passed\n";
}
