// Assert the compatibility contract through the real loader and result writer.
#include "core/plugin_manager.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

void runUtilityPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-utility-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create utility test folder");
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            fs::remove(folder / "utility_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::copy_file(pluginDirectory / "utility_plugin.dll", folder / "utility_plugin.dll");

    // An old host must reject the whole batch, retaining its original handler.
    check(registry.registerSimple("clean", "Old host fixture", "system",
        [](const std::string&) { CommandResult r; r.message = "old host"; return r; }, false),
        "Cannot register old host fixture");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 1 && !registry.isRegistered("help") &&
          registry.invoke("clean", "").message == "old host", "Plugin collision changed existing host behavior");
    registry.unregisterTool("clean");
    // Restart discovery to clear the rejected-file cache after removing the host fixture.
    PluginManager::unloadAll();
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 2, "Utility DLL did not publish two tools");
    for (const auto* id : {"clean", "help"}) {
        const auto tool = registry.getTool(id);
        const bool isHelp = std::string(id) == "help";
        check(tool && tool->provider_name == "Utility Plugin" &&
              tool->provider_type == GRIM::MMO::ToolProviderType::Plugin &&
              fs::path(tool->provider_path) == folder / "utility_plugin.dll" &&
              tool->permission_bits == 0 && tool->parameters.empty() && tool->examples.size() == 1 &&
              tool->is_informational == isHelp && tool->needs_confirmation != isHelp &&
              tool->category == (isHelp ? "information" : "system"), "Utility metadata changed policy or provider");
    }
    const std::string expectedHelp = "[Help] Available commands:\n"
        "- remember <key> <value>\n- recall <key>\n- forget <key>\n"
        "- pwd\n- cd <dir>\n- ls\n- mkdir <dir>\n- rm <file>\n"
        "- set_timer <seconds>\n- sysinfo\n- clean\n- help\n- voice\n- voice_stream\n";
    // Both original handlers ignore arguments. Compare every result field,
    // including voice/category/color, rather than only visible message text.
    for (const auto* args : {"", "ignored arguments"}) {
        const auto clean = registry.invoke("clean", args);
        check(clean.success && clean.message == "[Utility] Console cleared." && clean.errorCode == "ERR_NONE" &&
              clean.category == "routine" && clean.voice == "Console cleared" &&
              clean.color.toUInt() == Colors::Green.toUInt() && clean.structuredOutputJson.empty(),
              "clean result differs from original host handler");
        const auto help = registry.invoke("help", args);
        check(help.success && help.message == expectedHelp && help.errorCode == "ERR_NONE" &&
              help.category == "summary" && help.voice == "Help shown" &&
              help.color.toUInt() == Colors::Cyan.toUInt() && help.structuredOutputJson.empty(),
              "help result differs from static help with retired entries removed");
    }
    fs::remove(folder / "utility_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("clean") && !registry.isRegistered("help"),
          "Removing utility DLL retained registrations");
    std::cout << "Utility plugin result compatibility, collision, and folder removal tests passed\n";
}
