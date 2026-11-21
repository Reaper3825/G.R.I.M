// Real loader/API/registry reconciliation using tiny native fixtures.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "commands/command_result.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>

std::vector<std::string> pluginTestLogSnapshot();

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct ChangeObserver : GRIM::MMO::ToolRegistryListener {
    bool callable = false;
    void onToolRegistered(const std::string& id) override {
        callable = GRIM::MMO::ToolRegistry::instance().invoke(id, "").at("status") == "completed";
        throw std::runtime_error("observer failure");
    }
    void onToolUnregistered(const std::string&) override { throw std::runtime_error("observer failure"); }
    void onToolUpdated(const std::string&) override {}
};
}
void runPluginReconcileTests(const std::filesystem::path& variants, const std::filesystem::path& actual) {
    namespace fs = std::filesystem;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto watched = fs::absolute(variants / "watched");
    fs::create_directories(watched);
    // Only remove the exact test fixture destinations owned by this test.
    for (const auto* name : {"alpha.dll", "beta.dll", "alpha.tools.json", "beta.tools.json"}) fs::remove(watched / name);
    PluginManager::initialize(watched.string());
    PluginManager::checkForHotReload(); // Empty folders are valid.
    const auto install = [&](const char* variant) {
        fs::copy_file(variants / variant / "alpha.dll", watched / "alpha.dll", fs::copy_options::overwrite_existing);
        fs::copy_file(variants / variant / "alpha.tools.json", watched / "alpha.tools.json", fs::copy_options::overwrite_existing);
        PluginManager::checkForHotReload();
    };
    install("old");
    check(registry.toolCount() == baseline + 2, "Added file did not publish its complete batch");
    auto old = registry.getTool("Declared-alpha");
    check(old && !old->provider.instance.empty() && old->provider.path == (watched / "alpha.dll").string(),
          "Loaded generation/source identity missing");
    check(old->provider.version == "1.0" && old->provider.api_version == GRIM_API_VERSION, "Provider metadata did not come from the sidecar");
    const auto firstInstance = old->provider.instance;
    const auto provider = [&]() -> std::optional<GRIM::MMO::ProviderDescriptor> {
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if (group.provider.name == "Declared-alpha" && group.provider.type == GRIM::MMO::ToolProviderType::Plugin)
                return group.provider;
        return std::nullopt;
    };
    check(provider() && provider()->instance == firstInstance, "Plugin commit did not publish provider identity");
    PluginManager::initialize(watched.string());
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance, "Repeated initialize reloaded unchanged plugin");
    const auto resource = [](const char* kind) {
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if (group.provider.name == "Declared-alpha")
                for (const auto& item : group.resources)
                    if (item.kind == kind) return item.state.at("value").get<std::string>();
        return std::string{};
    };
    check(resource("fixture_init") == "old-result", "Init did not publish under the live provider");
    install("legacy");
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance, "Incompatible ABI displaced working plugin");
    PluginAPI::dispatchSystemTick(0, 0);
    check(resource("fixture_tick") == "old-result",
          "Live instance did not receive system ticks");
    install("failed");
    check(!provider() && registry.toolCount() == baseline && resource("fixture_init").empty(),
          "Failed init retained the provider, tools, or resources");
    const auto version = registry.version();
    PluginManager::checkForHotReload();
    check(registry.version() == version, "Unchanged rejected binary was repeatedly processed");
    install("old");
    install("collision");
    check(!provider() && registry.toolCount() == baseline, "Failed registration retained a provider");
    install("old");
    {
        std::ofstream broken(watched / "alpha.dll", std::ios::binary | std::ios::trunc);
        broken << "incomplete binary";
    }
    PluginManager::checkForHotReload();
    check(!provider(), "Failed native load retained the outgoing provider");
    install("old");
    const auto beforeCutover = pluginTestLogSnapshot().size();
    ChangeObserver notification;
    registry.addListener(&notification);
    const auto timestamp = fs::last_write_time(watched / "alpha.dll");
    fs::copy_file(variants / "new" / "alpha.dll", watched / "alpha.dll", fs::copy_options::overwrite_existing);
    fs::copy_file(variants / "new" / "alpha.tools.json", watched / "alpha.tools.json", fs::copy_options::overwrite_existing);
    fs::last_write_time(watched / "alpha.dll", timestamp);
    PluginManager::checkForHotReload();
    registry.removeListener(&notification);
    check(!notification.callable, "Registry observer entered the provider before init");
    const auto lifecycleLog = pluginTestLogSnapshot();
    const auto firstCutoverLog = lifecycleLog.begin() + beforeCutover;
    const auto shutdown = std::find(firstCutoverLog, lifecycleLog.end(), "[Plugin: Declared-alpha] Declared-alpha");
    const auto init = std::find(firstCutoverLog, lifecycleLog.end(), "[Plugin: Declared-alpha] init new-result");
    check(shutdown != lifecycleLog.end() && init != lifecycleLog.end() && shutdown < init,
          "Replacement init ran before outgoing shutdown");
    auto replacement = registry.getTool("Declared-alpha");
    check(provider() && replacement && provider()->instance == replacement->provider.instance,
          "Replacement provider and tools have different generations");
    check(replacement && replacement->provider.instance != firstInstance &&
          registry.invoke("Declared-alpha", "").at("message") == "new-result", "Valid replacement was not committed");
    check(!registry.isRegistered("Declared-alpha-extra"), "Replacement retained a removed tool");
    PluginAPI::dispatchSystemTick(0, 0);
    check(resource("fixture_tick") == "new-result",
          "Retiring old generation removed replacement resources");

    const auto secondInstance = replacement->provider.instance;
    check(registry.invoke("Declared-alpha", "reload").at("message") == "new-result", "Reload request did not return normally");
    check(registry.getTool("Declared-alpha")->provider.instance == secondInstance, "Plugin reloaded from inside its callback");
    PluginManager::tick();
    check(registry.getTool("Declared-alpha")->provider.instance != secondInstance, "Queued API reload request was not serviced");

    const auto beforeManifestEdit = registry.getTool("Declared-alpha")->provider.instance;
    const auto sidecar = watched / "alpha.tools.json";
    std::ofstream(sidecar, std::ios::app) << ' ';
    PluginManager::checkForHotReload();
    const auto afterManifestEdit = registry.getTool("Declared-alpha")->provider.instance;
    check(afterManifestEdit != beforeManifestEdit, "Manifest-only change did not reload the provider");
    std::ofstream(sidecar, std::ios::trunc) << '{';
    PluginManager::checkForHotReload();
    check(registry.getTool("Declared-alpha")->provider.instance == afterManifestEdit,
          "Invalid manifest displaced the working provider");
    fs::remove(sidecar);
    PluginManager::checkForHotReload();
    check(registry.getTool("Declared-alpha")->provider.instance == afterManifestEdit,
          "Missing manifest displaced the working provider");

    fs::create_directory(watched / "ignored.dll");
    PluginManager::checkForHotReload(); // A directory ending in .dll is not a plugin.
    fs::copy_file(variants / "new" / "alpha.dll", watched / "beta.dll");
    fs::copy_file(variants / "new" / "alpha.tools.json", watched / "beta.tools.json");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline + 1 &&
          registry.getTool("Declared-alpha")->provider.path == (watched / "alpha.dll").string(),
          "Duplicate provider replaced its live owner");
    fs::remove(watched / "alpha.dll"); // Source is unlocked while its private copy is loaded.
    PluginManager::checkForHotReload();
    check(registry.getTool("Declared-alpha")->provider.path == (watched / "beta.dll").string(),
          "Unchanged rejected plugin was not retried after its provider collision disappeared");
    fs::remove(watched / "beta.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline, "Deleted plugin retained tools");
    fs::remove(watched / "ignored.dll");
    fs::remove(watched / "alpha.tools.json");
    fs::remove(watched / "beta.tools.json");
    fs::remove(watched);
    PluginManager::checkForHotReload(); // Removal of the folder is a valid empty snapshot.
    fs::create_directory(watched);
    fs::copy_file(variants / "new" / "alpha.dll", watched / "alpha.dll");
    fs::copy_file(variants / "new" / "alpha.tools.json", watched / "alpha.tools.json");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    PluginManager::tick(); // Exercise the actual periodic frame path as well.
    check(registry.isRegistered("Declared-alpha"), "Plugin folder recreation was not discovered");
    PluginManager::unloadAll();
    PluginManager::tick();
    check(!provider(), "Provider identity survived plugin unload");
    check(registry.toolCount() == baseline, "Tick rediscovered plugins during teardown");
    fs::remove(watched / "alpha.dll");
    fs::remove(watched / "alpha.tools.json");
    fs::remove(watched / "beta.tools.json");
    fs::remove(watched);

    if (!actual.empty()) {
        PluginManager::initialize(fs::absolute(actual).string());
        const auto osint = registry.getTool("osint_report");
        const auto network = registry.getTool("networktest");
        check(osint && network && osint->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              osint->parameters.size() == 1 && !network->description.empty(), "Standalone plugin metadata did not load");
        check(registry.resolveAlias("osint") == std::optional<std::string>("osint_report"), "Standalone plugin alias missing");
        check(!registry.isRegistered("osint_show_ui"), "OSINT published a retired UI tool");
        PluginManager::unloadAll();
        check(registry.toolCount() == baseline, "Standalone unload retained tools");
    }
    std::cout << "Plugin folder reconciliation and replacement tests passed\n";
}
