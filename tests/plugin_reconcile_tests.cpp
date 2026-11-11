// Real loader/API/registry reconciliation using tiny native fixtures.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "commands/commands_core.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct ChangeObserver : GRIM::MMO::ToolRegistryListener {
    bool callable = false;
    void onToolRegistered(const std::string&) override { throw std::runtime_error("observer failure"); }
    void onToolUnregistered(const std::string&) override { throw std::runtime_error("observer failure"); }
    void onToolUpdated(const std::string& id) override {
        callable = dispatchCommand(id, "").message == "new-result";
        throw std::runtime_error("observer failure");
    }
};
}
void runPluginReconcileTests(const std::filesystem::path& variants, const std::filesystem::path& actual) {
    namespace fs = std::filesystem;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto watched = fs::absolute(variants / "watched");
    fs::create_directories(watched);
    // Only remove the exact test fixture destinations owned by this test.
    for (const auto* name : {"alpha.dll", "beta.dll"}) fs::remove(watched / name);
    PluginManager::initialize(watched.string());
    PluginManager::checkForHotReload(); // Empty folders are valid.
    const auto install = [&](const char* variant) {
        fs::copy_file(variants / variant / "alpha.dll", watched / "alpha.dll", fs::copy_options::overwrite_existing);
        PluginManager::checkForHotReload();
    };
    install("old");
    check(registry.toolCount() == baseline + 2, "Added file did not publish its complete batch");
    auto old = registry.getTool("Declared-alpha");
    check(old && !old->provider.instance.empty() && old->provider.path == (watched / "alpha.dll").string(),
          "Loaded generation/source identity missing");
    const auto firstInstance = old->provider.instance;
    const auto provider = [&]() -> std::optional<GRIM::MMO::ProviderDescriptor> {
        for (const auto& p : registry.snapshot().providers)
            if (p.name == "Declared-alpha" && p.type == GRIM::MMO::ToolProviderType::Plugin) return p;
        return std::nullopt;
    };
    check(provider() && provider()->instance == firstInstance, "Plugin commit did not publish provider identity");
    PluginManager::initialize(watched.string());
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance, "Repeated initialize reloaded unchanged plugin");
    auto* observer = PluginAPI::createPluginAPI("observer", "", 0);
    const auto context = [&](const char* key) {
        const auto* text = observer->get_context(observer, key);
        const auto result = text ? std::string(text) : std::string{};
        observer->string_free(observer, text);
        return result;
    };
    // A staged candidate's subscriptions/timers must not replace the old ones.
    install("failed");
    check(provider() && provider()->instance == firstInstance, "Failed init changed the published provider");
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance &&
          dispatchCommand("Declared-alpha", "").message == "old-result", "Failed init displaced working plugin");
    PluginAPI::emitGlobalEvent("fixture_event", nullptr, 0);
    PluginAPI::processTimers();
    check(context("fixture_event") == "old-result" && context("fixture_timer") == "old-result",
          "Failed candidate cleanup removed old resources or exposed staged callbacks");
    auto version = registry.version();
    PluginManager::checkForHotReload();
    check(registry.version() == version, "Unchanged rejected binary was repeatedly processed");

    install("collision");
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance, "Collision replaced working tools");
    install("legacy");
    check(registry.getTool("Declared-alpha")->provider.instance == firstInstance, "Incompatible ABI displaced working plugin");
    {
        std::ofstream broken(watched / "alpha.dll", std::ios::binary | std::ios::trunc);
        broken << "incomplete binary";
    }
    PluginManager::checkForHotReload();
    check(dispatchCommand("Declared-alpha", "").message == "old-result", "Invalid native library displaced working plugin");
    ChangeObserver notification;
    registry.addListener(&notification);
    const auto timestamp = fs::last_write_time(watched / "alpha.dll");
    fs::copy_file(variants / "new" / "alpha.dll", watched / "alpha.dll", fs::copy_options::overwrite_existing);
    fs::last_write_time(watched / "alpha.dll", timestamp);
    PluginManager::checkForHotReload();
    registry.removeListener(&notification);
    check(notification.callable, "Committed replacement was not callable from a registry observer");
    auto replacement = registry.getTool("Declared-alpha");
    check(provider() && replacement && provider()->instance == replacement->provider.instance,
          "Replacement provider and tools have different generations");
    check(replacement && replacement->provider.instance != firstInstance &&
          dispatchCommand("Declared-alpha", "").message == "new-result", "Valid replacement was not committed");
    check(!registry.isRegistered("Declared-alpha-extra"), "Replacement retained a removed tool");
    PluginAPI::emitGlobalEvent("fixture_event", nullptr, 0);
    PluginAPI::processTimers();
    check(context("fixture_event") == "new-result" && context("fixture_timer") == "new-result",
          "Retiring old generation removed replacement resources");

    const auto secondInstance = replacement->provider.instance;
    check(dispatchCommand("Declared-alpha", "reload").message == "new-result", "Reload request did not return normally");
    check(registry.getTool("Declared-alpha")->provider.instance == secondInstance, "Plugin reloaded from inside its callback");
    PluginManager::tick();
    check(registry.getTool("Declared-alpha")->provider.instance != secondInstance, "Queued API reload request was not serviced");

    fs::create_directory(watched / "ignored.dll");
    PluginManager::checkForHotReload(); // A directory ending in .dll is not a plugin.
    fs::copy_file(variants / "new" / "alpha.dll", watched / "beta.dll");
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
    fs::remove(watched);
    PluginManager::checkForHotReload(); // Removal of the folder is a valid empty snapshot.
    fs::create_directory(watched);
    fs::copy_file(variants / "new" / "alpha.dll", watched / "alpha.dll");
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    PluginManager::tick(); // Exercise the actual periodic frame path as well.
    check(registry.isRegistered("Declared-alpha"), "Plugin folder recreation was not discovered");
    PluginManager::unloadAll();
    PluginManager::tick();
    check(!provider(), "Provider identity survived plugin unload");
    check(registry.toolCount() == baseline, "Tick rediscovered plugins during teardown");
    fs::remove(watched / "alpha.dll");
    fs::remove(watched);
    PluginAPI::cleanupPluginContext(observer);

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
