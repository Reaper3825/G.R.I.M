// Real DLL -> PluginManager -> API -> ToolRegistry -> filesystem, in a private sandbox.
#include "core/plugin_manager.hpp"
#include "MMO/Core/ToolRegistry.hpp"
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
    const auto sandbox = fs::temp_directory_path() / ("grim-filesystem-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(sandbox), "Cannot create private filesystem test directory");
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            // Only known fixture paths are removed. No recursive cleanup.
            for (const auto* name : {"plugins/filesystem_plugin.dll", "sample.txt", "new folder", "nonempty/keep.txt", "nonempty", "plugins"})
                fs::remove(path / name, error);
            fs::remove(path, error);
        }
    } cleanup{sandbox};
    fs::create_directory(sandbox / "plugins");
    const auto deployed = sandbox / "plugins/filesystem_plugin.dll";
    fs::copy_file(pluginDirectory / "filesystem_plugin.dll", deployed);
    PluginManager::initialize((sandbox / "plugins").string());
    check(registry.toolCount() == baseline + 5, "Filesystem DLL did not register exactly five tools");
    for (const auto* id : {"pwd", "cd", "ls", "mkdir", "rm"}) {
        const auto tool = registry.getTool(id);
        check(tool && tool->provider_type == GRIM::MMO::ToolProviderType::Plugin &&
              tool->provider_name == "Filesystem Plugin" && fs::path(tool->provider_path) == deployed,
              "Filesystem tool has incorrect provider ownership");
        check(!tool->usage.empty() && tool->examples.size() == 1 && tool->permission_bits == GRIM_PERM_FILESYSTEM,
              "Filesystem tool metadata incomplete");
        const bool informational = std::string(id) == "pwd" || std::string(id) == "ls";
        check(tool->is_informational == informational && tool->needs_confirmation != informational &&
              tool->parameters.size() == (informational ? 0 : 1), "Filesystem parameter/policy metadata incorrect");
    }
    const auto invoke = [&](const char* id, const std::string& args = "") { return registry.invoke(id, args); };
    check(invoke("pwd").message == "[FS] Current directory: " + processDirectory.string(), "Initial directory incorrect");
    check(invoke("cd", sandbox.string()).success, "Absolute cd failed");
    check(fs::current_path() == processDirectory, "Plugin cd changed host process directory");
    check(invoke("mkdir", "\"new folder\"").success && fs::is_directory(sandbox / "new folder"), "Quoted mkdir failed");
    check(!invoke("mkdir", "new folder").success, "Duplicate mkdir succeeded");
    check(!invoke("mkdir", "missing/child").success, "mkdir created missing parents");
    { std::ofstream file(sandbox / "sample.txt"); file << "fixture"; }
    check(invoke("ls").message.find("sample.txt") != std::string::npos, "ls omitted file");
    check(invoke("cd", "sample.txt").errorCode == "ERR_FS_NOT_DIRECTORY", "cd accepted regular file");
    check(invoke("cd", "missing").errorCode == "ERR_FS_NOT_FOUND", "Missing cd path not reported");
    check(invoke("pwd").message == "[FS] Current directory: " + fs::canonical(sandbox).string(), "Failed cd changed directory");
    check(invoke("cd", "new folder").success && invoke("cd", "..").success, "Relative cd failed");
    for (const auto* id : {"cd", "mkdir", "rm"})
        check(invoke(id).errorCode == "ERR_FS_NO_ARGUMENT", "Missing argument not rejected");
    check(invoke("rm", std::string("bad\0path", 8)).errorCode == "ERR_FS_INVALID_PATH", "NUL path accepted");
    fs::create_directory(sandbox / "nonempty");
    { std::ofstream file(sandbox / "nonempty/keep.txt"); file << "keep"; }
    check(!invoke("rm", "nonempty").success && fs::exists(sandbox / "nonempty/keep.txt"), "rm removed nonempty directory");
    check(invoke("rm", "sample.txt").success && !fs::exists(sandbox / "sample.txt"), "rm file failed");
    check(invoke("rm", "new folder").success && !fs::exists(sandbox / "new folder"), "rm empty directory failed");
    check(invoke("rm", "missing").errorCode == "ERR_FS_NOT_FOUND", "Missing rm path not reported");
    fs::remove(deployed);
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline, "Removing filesystem DLL retained tools");
    check(invoke("pwd").errorCode == "ERR_UNKNOWN_CMD", "Removed filesystem tool still executable");
    std::cout << "Filesystem plugin metadata, operations, and folder removal tests passed\n";
}
