#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/ToolDescriptorJson.hpp"
#include "commands/command_result.hpp"
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
        GrimPluginAPI* api = nullptr;
        ~Cleanup() {
            PluginManager::unloadAll();
            ToolRegistry::instance().unregisterByProvider("inspection-fixtures", ToolProviderType::Builtin);
            if (api) PluginAPI::cleanupPluginContext(api);
            std::error_code error;
            fs::remove(folder / "tool_inspection_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    cleanup.api = PluginAPI::createPluginAPI("snapshot-reader", "", 0);
    auto* api = cleanup.api;
    const auto snapshot = [&](const char* id = nullptr) {
        const auto* text = api->get_tool_snapshot(api, id);
        check(text != nullptr, "Snapshot API failed");
        const std::string copy(text);
        api->string_free(api, text);
        const auto value = nlohmann::json::parse(copy);
        check(value.at("format") == "grim-tool-snapshot" && value.at("version") == 1, "Snapshot schema missing");
        return value.at("tools");
    };
    check(snapshot().size() == baseline && snapshot("missing").empty(), "Empty/missing snapshot semantics incorrect");
    check(api->get_tool_snapshot(nullptr, nullptr) == nullptr, "Invalid API identity accepted");
    fs::copy_file(pluginDirectory / "tool_inspection_plugin.dll", folder / "tool_inspection_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 3, "Inspection plugin did not register three tools");
    for (const auto* id : {"list_tools", "tool_info", "tool_stats"}) {
        const auto d = registry.getTool(id);
        check(d && d->provider_name == "Tool Inspection Plugin" && d->provider_type == ToolProviderType::Plugin &&
              d->is_informational && !d->needs_confirmation && d->permission_bits == 0 && d->parameters.size() == 1,
              "Inspection tool metadata/provider/policy incorrect");
    }
    const auto invoke = [&](const char* id, const std::string& args = "") { return registry.invoke(id, args); };
    const auto matches = [&](const CommandResult& r, bool ok, const std::string& text,
                              const char* code, const char* voice) {
        check(r.success == ok && r.message == text && r.errorCode == code && r.voice == voice &&
              r.category == (ok ? "information" : "error") && r.structuredOutputJson.empty() &&
              r.color.toUInt() == (ok ? Colors::Cyan : Colors::Red).toUInt(), "Inspection result compatibility failed");
    };
    matches(invoke("tool_stats"), false, "[Error] No usage statistics available", "ERR_NO_STATS", "No statistics");
    matches(invoke("tool_info"), false, "[Error] Usage: tool_info <command_name>", "ERR_NO_ARGUMENT", "Please specify a tool name");
    matches(invoke("tool_info", "missing"), false, "[Error] Tool not found: missing", "ERR_NOT_FOUND", "Tool not registered");
    matches(invoke("list_tools", "missing"), false, "[Error] No tools in category: missing", "ERR_NO_CATEGORY", "Category not found");

    ToolDescriptor rich;
    rich.tool_id = "inspection_fixture";
    rich.display_name = "Fixture Display";
    rich.provider_name = "inspection-fixtures";
    rich.provider_instance = "fixture-generation";
    rich.provider_path = "C:/fixture.dll";
    rich.version = "9.8.7";
    rich.description = "Fixture description";
    rich.usage = "inspection_fixture <value>";
    rich.category = "fixture";
    rich.is_informational = false;
    rich.needs_confirmation = true;
    rich.permission_bits = GRIM_PERM_FILESYSTEM;
    rich.argument_format = "json";
    rich.argument_schema_json = "{\"type\":\"object\"}";
    rich.parameters = {{"value", "string", "Value description", true}, {"optional", "number", "Optional value", false}};
    rich.examples = {"inspection_fixture demo"};
    rich.aliases = {"fixture_alias"};
    rich.keywords = {"first", "second"};
    rich.preconditions = {"ready"};
    rich.context_requirements = {"context"};
    rich.affordance_tokens = {"<fixture>"};
    rich.capability_tags = {"tag"};
    rich.usage_count = 100;
    rich.success_count = 75;
    rich.failure_count = 25;
    rich.success_rate = 0.75f;
    int fixtureCalls = 0;
    registry.registerTool(rich, [&](const std::string&) { ++fixtureCalls; return CommandResult{}; });
    const auto encoded = snapshot("fixture_alias");
    check(encoded.size() == 1 && encoded[0].size() == 28 && encoded[0].at("provider_instance") == "fixture-generation" &&
          encoded[0].at("argument_schema_json") == rich.argument_schema_json && encoded[0].at("success_count") == 75 &&
          encoded[0].at("context_requirements")[0] == "context" && encoded[0].at("parameters")[0].at("required") == true,
          "Snapshot omitted rich metadata");
    check(snapshot("Fixture Display") == encoded && snapshot("inspection_fixture") == encoded &&
          registry.getTool(rich.tool_id)->usage_count == 100 && fixtureCalls == 0, "Snapshot lookup mutated or invoked tool");
    const std::string info = "[Tool: inspection_fixture]\n\nDescription: Fixture description\nCategory   : fixture\n"
        "Type       : Action\nUsage      : inspection_fixture <value>\nAliases    : fixture_alias\nKeywords   : first, second\n"
        "\nParameters:\n  value (string) [required]\n    Value description\n  optional (number)\n    Optional value\n"
        "\nExamples:\n  - inspection_fixture demo\n\nUsage Stats:\n  Total uses  : 100\n  Success rate: 75%\n";
    matches(invoke("tool_info", "fixture_alias"), true, info, "ERR_NONE", "Tool information displayed");
    matches(invoke("tool_info", "Fixture Display"), true, info, "ERR_NONE", "Tool information displayed");
    matches(invoke("list_tools", "fixture"), true, "[fixture tools - 1 total]\n\n  inspection_fixture: Fixture description\n", "ERR_NONE", "Tool list displayed");
    matches(invoke("list_tools", " fixture"), false, "[Error] No tools in category:  fixture", "ERR_NO_CATEGORY", "Category not found");
    const auto all = invoke("list_tools");
    check(all.success && all.message.find("[Registered Tools - " + std::to_string(baseline + 4) + " total]") == 0 &&
          all.message.find("  inspection_fixture (aliases: fixture_alias)\n    Fixture description\n") != std::string::npos,
          "All-tools grouping/aliases/count changed");
    const std::string stats = "[Tool Usage Statistics]\n\n1. inspection_fixture\n   Uses: 100 (Success: 75, Fail: 25)\n   Success Rate: 75%\n\n";
    matches(invoke("tool_stats", "1"), true, stats, "ERR_NONE", "Statistics displayed");
    matches(invoke("tool_stats", "1suffix"), true, stats, "ERR_NONE", "Statistics displayed");
    matches(invoke("tool_stats", "-1"), true, "[Tool Usage Statistics]\n\n", "ERR_NONE", "Statistics displayed");
    matches(invoke("tool_stats", "0"), true, "[Tool Usage Statistics]\n\n", "ERR_NONE", "Statistics displayed");
    check(invoke("tool_stats", "invalid").errorCode == "ERR_CMD_EXCEPTION" &&
          invoke("tool_stats", "99999999999999999999").errorCode == "ERR_CMD_EXCEPTION", "Invalid statistics parsing changed");
    rich.description = std::string(70000, 'x');
    rich.swap_state = ToolSwapState::Unavailable;
    registry.updateMetadata(rich);
    check(snapshot("fixture_alias")[0].at("description").get<std::string>().size() == 70000 &&
          snapshot("fixture_alias")[0].at("swap_state") == static_cast<int>(ToolSwapState::Unavailable) &&
          encoded[0].at("description") == "Fixture description" && fixtureCalls == 0,
          "Snapshot truncated data, cached stale metadata, or lost copy isolation");
    const auto largeInfo = invoke("tool_info", "fixture_alias");
    const auto largeList = invoke("list_tools", "fixture");
    check(largeInfo.success && largeInfo.message.find(rich.description) != std::string::npos &&
          largeList.success && largeList.message.find(rich.description) != std::string::npos && fixtureCalls == 0,
          "Inspection result was truncated or rejected at the old 64-KiB message limit");
    registry.unregisterTool(rich.tool_id);
    check(snapshot("fixture_alias").empty(), "Removed alias retained stale snapshot");
    fs::remove(folder / "tool_inspection_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("list_tools") && !registry.isRegistered("tool_info") &&
          !registry.isRegistered("tool_stats"), "Removing inspection DLL retained tools");
    std::cout << "Tool inspection plugin snapshot, compatibility, statistics, and removal tests passed\n";
}
