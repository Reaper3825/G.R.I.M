// Exercise discovery, initialization, publication, and shutdown with native fixture DLLs.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

struct BatchObserver : GRIM::MMO::ToolRegistryListener {
    bool complete = false;
    void onToolRegistered(const std::string&) override {
        auto& tools = GRIM::MMO::ToolRegistry::instance();
        const auto providers = GRIM::MMO::ProviderRegistry::instance().snapshot();
        complete = tools.invoke("Declared-alpha", "").at("status") == "completed" &&
                   tools.isRegistered("Declared-alpha-extra") &&
                   providers.resourceCount() == 2;
    }
    void onToolUnregistered(const std::string&) override {}
    void onToolUpdated(const std::string&) override {}
};
}

void runPluginBatchTests(const std::filesystem::path& variants, const std::filesystem::path& actual) {
    namespace fs = std::filesystem;
    using Status = PluginBatchResult::Status;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto root = fs::absolute(variants / "batch-fixtures");
    const auto install = [&](const char* variant, const std::string& stem, const char* provider = nullptr, bool keepTools = false) {
        const auto folder = root / stem;
        fs::create_directories(folder);
        fs::copy_file(variants / variant / "alpha.dll", folder / (stem + ".dll"), fs::copy_options::overwrite_existing);
        auto manifest = nlohmann::json::parse(std::ifstream(variants / variant / "alpha.tools.json"));
        if (provider) {
            manifest["provider"] = provider;
            if (!keepTools) manifest["tools"] = nlohmann::json::array();
        }
        std::ofstream(folder / (stem + ".tools.json")) << manifest;
    };
    install("old", "alpha");
    install("new", "beta", "Batch beta");
    install("new", "collision", "Batch collision", true);
    install("failed", "failed", "Batch failed");
    install("legacy", "legacy", "Batch legacy");
    install("new", "missing", "Batch missing manifest");
    fs::remove(root / "missing" / "missing.tools.json");
    install("new", "zduplicate");
    constexpr const char* invalidName = "invalid";
    install("new", invalidName, "Batch invalid");
    std::ofstream(root / invalidName / "invalid.dll", std::ios::binary | std::ios::trunc) << "not a native library";

    BatchObserver observer;
    registry.addListener(&observer);
    const auto before = registry.version();
    const auto result = PluginManager::loadBatch(root);
    registry.removeListener(&observer);
    check(result.status == Status::Completed && result.plugins.size() == 8, "Discovery omitted batch outcomes");
    using Outcome = PluginLoadStatus;
    const Outcome expected[]{Outcome::Loaded, Outcome::Loaded, Outcome::RegistrationRejected, Outcome::InitializationFailed,
        Outcome::NativeLoadFailed, Outcome::InvalidBundle, Outcome::InvalidBundle,
        Outcome::RegistrationRejected};
    for (std::size_t i = 0; i < result.plugins.size(); ++i)
        check(result.plugins[i].status == expected[i], "Plugin load returned the wrong outcome");
    check(result.plugins[6].detail.find("missing.tools.json") != std::string::npos, "Missing manifest rejection lacked context");
    check(observer.complete && registry.version() == before + 1, "Publication exposed an incomplete batch");
    check(registry.toolCount() == baseline + 2, "Rejected plugin published tools");
    auto& providers = GRIM::MMO::ProviderRegistry::instance();
    const auto startup = providers.startupProviders();
    check(startup.size() == 2 && startup[1].name == "Batch beta",
          "Startup roster omitted the tool-less provider or included a rejected provider");
    const auto instance = registry.getTool("Declared-alpha")->provider.instance;
    check(PluginManager::loadBatch(root).status == Status::AlreadyStarted, "Startup batch ran twice");
    PluginAPI::dispatchSystemTick(0, 0);
    check(GRIM::MMO::ProviderRegistry::instance().snapshot().resourceCount() == 4,
          "Ready plugins did not receive ticks or a rejected plugin published resources");
    check(PluginManager::shutdownBatch() == 2 && registry.version() == before + 2,
          "Shutdown did not remove providers in one batch");
    PluginAPI::dispatchSystemTick(0, 0);
    check(PluginManager::shutdownBatch() == 0 && registry.toolCount() == baseline &&
          GRIM::MMO::ProviderRegistry::instance().snapshot().resourceCount() == 0,
          "Shutdown retained tools or resources");

    install("new", "alpha");
    PluginManager::loadBatch(root);
    check(registry.getTool("Declared-alpha")->provider.instance != instance &&
          registry.invoke("Declared-alpha", "").at("message") == "new-result",
          "New host lifecycle retained a stale plugin instance");
    std::vector<GRIM::MMO::ProviderUpdate> updates;
    auto feed = providers.subscribeProviderUpdates(providers.startupProviders(), [&](auto batch) {
        updates = std::move(batch);
    });
    auto tool = registry.getTool("Declared-alpha").value();
    tool.description = "Updated fixture description";
    check(registry.updateMetadata(tool), "Fixture tool metadata update failed");
    providers.dispatchProviderUpdates();
    check(updates.size() == 1 && updates.front().providerId == tool.provider.instance &&
          updates.front().tools && updates.front().tools->front().description == tool.description &&
          !updates.front().resources, "Tool metadata did not publish by provider ID");
    check(registry.unregisterTool(tool.tool_id, tool.provider.name, tool.provider.instance),
          "Fixture tool withdrawal failed");
    providers.dispatchProviderUpdates();
    check(updates.size() == 1 && updates.front().providerId == tool.provider.instance &&
          updates.front().tools && updates.front().tools->empty(), "Tool withdrawal did not clear provider tools");
    PluginManager::shutdownBatch();
    for (const auto& entry : fs::directory_iterator(root)) {
        const auto folder = entry.path();
        const auto stem = folder.filename().string();
        fs::remove(folder / (stem + ".dll"));
        fs::remove(folder / (stem + ".tools.json"));
        fs::remove(folder);
    }
    fs::remove(root);
    if (!actual.empty()) {
        PluginManager::loadBatch(fs::absolute(actual).string());
        const auto osint = registry.getTool("osint_report");
        const auto network = registry.getTool("networktest");
        check(osint && network && osint->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              osint->parameters.size() == 1 && !network->description.empty(), "Standalone plugin metadata did not load");
        check(registry.resolveAlias("osint") == std::optional<std::string>("osint_report"), "Standalone plugin alias missing");
        check(!registry.isRegistered("osint_show_ui"), "OSINT published a retired UI tool");
        PluginManager::shutdownBatch();
        check(registry.toolCount() == baseline, "Standalone unload retained tools");
    }
    std::cout << "Plugin startup and shutdown batch tests passed\n";
}
