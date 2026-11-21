#include "core/plugin_manager.hpp"
#include "plugin_deployment_fixture.hpp"
#include "plugin_batch_fixture.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

void runToolInspectionPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-tool-inspection-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create inspection test folder");
    struct Cleanup {
        fs::path folder;
        std::unique_ptr<PluginBatchFixture> api;
        ~Cleanup() {
            PluginManager::shutdownBatch();
            api.reset();
            std::error_code error;
            fs::remove(folder / "tool_inspection_plugin/tool_inspection_plugin.dll", error);
            fs::remove(folder / "tool_inspection_plugin/.deployment", error);
            fs::remove(folder / "tool_inspection_plugin/tool_inspection_plugin.tools.json", error);
            fs::remove(folder / "tool_inspection_plugin", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    cleanup.api = std::make_unique<PluginBatchFixture>(GRIM::MMO::ProviderDescriptor{.name = "snapshot-reader", .path = ""});
    cleanup.api->load();
    auto& api = *cleanup.api;
    const auto snapshot = [&](const char* id = nullptr) {
        const auto* text = api->get_tool_snapshot(api.handle(), id);
        check(text != nullptr, "Snapshot API failed");
        const std::string copy(text);
        api->string_free(api.handle(), text);
        const auto value = nlohmann::json::parse(copy);
        check(value.at("format") == "grim-tool-snapshot" && value.at("version") == 1, "Snapshot schema missing");
        return value.at("tools");
    };
    check(snapshot().size() == baseline && snapshot("missing").empty(), "Empty/missing snapshot semantics incorrect");
    check(api->get_tool_snapshot(0, nullptr) == nullptr, "Invalid API identity accepted");
    fs::create_directory(folder / "tool_inspection_plugin");
    fs::copy_file(pluginDirectory / "tool_inspection_plugin.dll", folder / "tool_inspection_plugin/tool_inspection_plugin.dll");
    fs::copy_file(pluginDirectory / "tool_inspection_plugin.tools.json", folder / "tool_inspection_plugin/tool_inspection_plugin.tools.json");
    writePluginFixtureRevision(folder / "tool_inspection_plugin");
    PluginManager::loadBatch(folder.string());
    check(registry.toolCount() == baseline + 3, "Inspection plugin did not register three tools");
    for (const auto* id : {"list_tools", "tool_info", "tool_stats"}) {
        const auto d = registry.getTool(id);
        check(d && d->provider.name == "Tool Inspection Plugin" && d->provider.type == ToolProviderType::Plugin &&
              d->is_informational && !d->needs_confirmation && d->parameters.size() == 1,
              "Inspection tool metadata/provider/policy incorrect");
    }
    const auto invoke = [&](const char* id, const std::string& args = "") { return registry.invoke(id, args); };
    const auto matches = [&](const nlohmann::json& r, bool ok, const std::string& text,
                              const char* code) {
        check((r.at("status") == "completed") == ok && r.at("message") == text && r.at("error_code") == code &&
              r.at("category") == (ok ? "information" : "error") && r.at("payload").empty(), "Inspection result compatibility failed");
    };
    matches(invoke("tool_stats"), false, "[Error] No usage statistics available", "ERR_NO_STATS");
    matches(invoke("tool_info"), false, "[Error] Usage: tool_info <command_name>", "ERR_NO_ARGUMENT");
    matches(invoke("tool_info", "missing"), false, "[Error] Tool not found: missing", "ERR_NOT_FOUND");
    matches(invoke("list_tools", "missing"), false, "[Error] No tools in category: missing", "ERR_NO_CATEGORY");

    PluginManager::shutdownBatch();
    fs::remove(folder / "tool_inspection_plugin/tool_inspection_plugin.dll");
    fs::remove(folder / "tool_inspection_plugin/.deployment");
    fs::remove(folder / "tool_inspection_plugin/tool_inspection_plugin.tools.json");
    fs::remove(folder / "tool_inspection_plugin");
    check(registry.toolCount() == baseline && !registry.isRegistered("list_tools") && !registry.isRegistered("tool_info") &&
          !registry.isRegistered("tool_stats"), "Removing inspection DLL retained tools");
    std::cout << "Tool inspection plugin snapshot, compatibility, and removal tests passed\n";
}
