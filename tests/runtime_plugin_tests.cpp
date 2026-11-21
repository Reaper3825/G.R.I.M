// Real memory provider with a private persistence directory.
#include "core/plugin_manager.hpp"
#include "commands/command_result.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void runRuntimePluginTests(const std::filesystem::path& memoryDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    using nlohmann::json;
    if (memoryDirectory.empty()) return;

    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-runtime-plugins-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto bundle = folder / "memory_plugin";
    const auto stateDirectory = bundle / "memory_plugin_data";
    check(fs::create_directories(stateDirectory), "Cannot create runtime plugin fixture");
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll();
            std::error_code error;
            fs::remove_all(folder, error);
        }
    } cleanup{folder};

    // Seed a short-term record without adding a production write endpoint for tests.
    const json shortTerm{{"id", 101}, {"raw", "recent blue fact"}, {"tags", {"recent"}},
                         {"embedding", std::vector<float>(768, 0.25f)}};
    std::ofstream snapshot(stateDirectory / "transient.json");
    snapshot << json{{"version", 1}, {"pending", json::array()}, {"short_term", json::array({shortTerm})}};
    snapshot.close();
    check(!snapshot.fail(), "Cannot seed short-term memory");

    const auto recall = [&](const char* query) { return registry.invoke("recall", query); };
    const auto retrieve = [&](const char* prompt, const char* category) {
        const auto result = registry.invoke("memory.retrieve",
            json{{"prompt", prompt}, {"nlp_category", category}, {"limit", 10}}.dump());
        check(result.at("status") == "completed", "Provider retrieval failed");
        return result.at("payload");
    };

    fs::copy_file(memoryDirectory / "memory_plugin.dll", bundle / "memory_plugin.dll");
    fs::copy_file(memoryDirectory / "memory_plugin.tools.json", bundle / "memory_plugin.tools.json");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 3, "Memory provider registration count differs");
    for (const auto* id : {"remember", "recall"}) {
        const auto tool = registry.getTool(id);
        const bool informational = std::string_view(id) == "recall";
        check(tool && tool->provider.name == "Memory Plugin" &&
              tool->is_informational == informational && tool->needs_confirmation != informational,
              "Memory tool metadata differs");
    }
    const auto retrievalTool = registry.getTool("memory.retrieve");
    check(retrievalTool && retrievalTool->usage_type == ToolUsage::System, "Retrieval is not classified as a System tool");
    const auto modelTools = registry.getModelTools();
    check(std::none_of(modelTools.begin(), modelTools.end(),
        [](const auto& tool) { return tool.tool_id == "memory.retrieve"; }), "Retrieval leaked into model discovery");
    check(!registry.isRegistered("forget"), "Retired forget tool was registered");
    check(registry.invoke("remember", "").at("error_code") == "ERR_MEMORY_MISSING_INPUT", "Empty remember was accepted");
    check(recall("").at("error_code") == "ERR_MEMORY_MISSING_KEY", "Empty recall was accepted");
    check(registry.invoke("memory.retrieve", R"({"prompt":"blue","nlp_category":"","limit":0})").at("status") == "failed",
          "Invalid retrieval limit was accepted");

    const auto remembered = registry.invoke("remember", "my color is blue");
    check(remembered.at("status") == "completed" && remembered.at("message") == "[Memory] Remembered: my color is blue", "Remember failed");
    check(recall("my color").at("status") == "failed", "Remembered record was committed before retrieval");
    check(recall("blue").at("message") == "[Memory] recent blue fact", "Short-term recall failed");

    PluginManager::unloadAll();
    check(registry.toolCount() == baseline, "Unloaded memory provider retained tools");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 3, "Memory provider did not reload");
    check(recall("my color").at("status") == "failed", "Reload changed the pending record to long-term");
    check(recall("blue").at("message") == "[Memory] recent blue fact", "Reload lost short-term memory");

    const auto copied = retrieve("blue", "manual");
    const auto& hits = copied.at("memories");
    check(hits.size() == 2 && hits[0].at("id") == 101 &&
          hits[1].at("raw") == "my color is blue" &&
          hits[1].at("normalized") == "remember my color is blue" &&
          hits[1].at("confidence") == json(0.98f) && hits[1].at("score") == json(0.98f) &&
          hits[1].at("tags") == json({"manual", "remember"}), "Retrieval lost ordering or manual record fields");
    check(copied.at("retrieval_breadcrumbs") == json::array({
        {{"query", "blue"}, {"results_found", 2}},
        {{"query", "tag:manual"}, {"results_found", 1}}
    }), "Retrieval breadcrumbs or tag deduplication differ");
    check(recall("my color").at("message") == "[Memory] my color is blue", "Pending memory was not promoted");
    check(recall("BLUE").at("error_code") == "ERR_MEMORY_KEY_NOT_FOUND", "Recall case sensitivity changed");

    PluginManager::unloadAll();
    check(hits[0].at("raw") == "recent blue fact", "Copied results were invalidated by unload");
    std::ifstream saved(stateDirectory / "transient.json");
    const auto transient = json::parse(saved);
    check(transient.at("pending").empty() && transient.at("short_term").at(0).at("embedding").at(0) == 0.25,
          "Transient snapshot lost state");
    saved.close();
    PluginManager::initialize(folder.string());
    check(retrieve("blue", "").at("memories") == hits, "Committed or short-term memory did not survive a second load");
    PluginManager::unloadAll();
    check(registry.toolCount() == baseline, "Runtime plugin cleanup left registrations");
    std::cout << "Memory provider: ownership, copied retrieval, persistence and reload tests passed\n";
}
