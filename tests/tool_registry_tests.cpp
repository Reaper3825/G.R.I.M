#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/ToolDescriptorJson.hpp"
#include "MMO/Core/ToolGapPlanner.hpp"
#include "commands/commands_core.hpp"
#include "core/plugin.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

void ensureCorePluginsRegistered() {}
void logError(const std::string&, const std::string&) {}
namespace {
using namespace GRIM::MMO;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
ToolDescriptor descriptor(const std::string& id, const std::string& provider = "tests") {
    ToolDescriptor tool;
    tool.tool_id = tool.display_name = id;
    tool.provider_name = provider;
    tool.provider_type = ToolProviderType::Plugin;
    tool.category = "test";
    return tool;
}
CommandResult echo(const std::string& argument) {
    CommandResult result;
    result.message = argument;
    return result;
}
struct Observer : ToolRegistryListener {
    int changes = 0;
    void onToolRegistered(const std::string& id) override {
        check(ToolRegistry::instance().isRegistered(id), "Listener saw uncommitted registration");
        ++changes;
    }
    void onToolUnregistered(const std::string& id) override {
        check(!ToolRegistry::instance().isRegistered(id), "Listener saw retained registration");
        ++changes;
    }
    void onToolUpdated(const std::string& id) override {
        check(ToolRegistry::instance().isRegistered(id), "Listener lost updated tool");
        ++changes;
    }
};
}

int main() try {
    using namespace GRIM::MMO;
    // Labels survive snapshots and never resolve or convert handler arguments.
    {
        auto labeled = descriptor("binding_fixture");
        labeled.usage = "binding_fixture {target}";
        labeled.short_description = "Inspect information supplied by a selected entity";
        labeled.bindings = {{"target", "t"}};
        const auto json = nlohmann::json(labeled);
        check(json.get<ToolDescriptor>().bindings == labeled.bindings, "Binding snapshot roundtrip failed");
        check(json.get<ToolDescriptor>().short_description == labeled.short_description, "Short summary snapshot roundtrip failed");
        auto legacySummary = json; legacySummary.erase("short_description");
        check(legacySummary.get<ToolDescriptor>().short_description.empty(), "Legacy summary default changed");
        for (const auto* invalid : {"Only four words here", "one two three four five six seven eight nine ten eleven twelve thirteen", "Five words with\na newline"}) {
            auto bad = json; bad["short_description"] = invalid;
            bool rejected = false;
            try { bad.get<ToolDescriptor>(); } catch (const std::exception&) { rejected = true; }
            check(rejected, "Invalid catalog summary accepted");
        }
        auto old = json; old.erase("bindings");
        check(old.get<ToolDescriptor>().bindings.empty(), "Old snapshot did not default to empty bindings");
        for (const auto& bad : nlohmann::json::array({
                {{"target", "interface"}}, {{"missing", "t"}}, {{"", "t"}}})) {
            auto malformed = json; malformed["bindings"] = bad;
            bool rejected = false;
            try { malformed.get<ToolDescriptor>(); } catch (const std::exception&) { rejected = true; }
            check(rejected, "Invalid binding snapshot accepted");
        }
        auto& registry = ToolRegistry::instance();
        registry.registerTool(labeled, echo);
        check(registry.invoke(labeled.tool_id, "an unresolved entity").message == "an unresolved entity",
              "Binding labels changed handler arguments");
        check(registry.generateFullPrompt().find("Usage: binding_fixture {target}") != std::string::npos &&
              registry.generateFullPrompt().find("Binding:") == std::string::npos,
              "Full prompt lost usage or exposed system-facing bindings");
        labeled.bindings = {{"target", "new_type"}};
        check(!registry.updateMetadata(labeled), "Invalid metadata binding accepted");
        labeled.tool_id = "invalid_binding";
        bool rejected = false;
        try { registry.registerTool(labeled, echo); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected && !registry.isRegistered(labeled.tool_id), "Invalid registration binding accepted");
        registry.unregisterTool("binding_fixture");
    }
    auto& registry = ToolRegistry::instance();
    Observer observer;
    registry.addListener(&observer);
    auto first = descriptor("first");
    first.aliases = {"alias"};
    check(registry.registerTool(first, echo), "Registration failed");
    check(registry.toolCount() == 1 && registry.resolveAlias("alias") == "first", "Metadata and callable not published together");
    check(dispatchCommand("alias", "argument").message == "argument", "Alias used a different execution store");
    check(registry.getTool("first")->usage_count == 1 && registry.getTool("alias")->success_count == 1, "Dispatch did not count exactly once on canonical tool");
    const auto version = registry.version();
    check(!registry.registerTool(first, echo), "Duplicate replaced tool implicitly");
    auto collision = descriptor("second");
    collision.aliases = {"alias"};
    check(!registry.registerTool(collision, echo), "Alias collision accepted");
    check(!registry.registerTool(descriptor("alias"), echo), "Canonical name overwrote alias");
    check(registry.version() == version && registry.toolCount() == 1, "Rejected registrations mutated registry");
    auto foreign = first;
    foreign.provider_name = "other";
    check(!registry.registerTool(foreign, echo, true), "Provider replaced foreign tool");
    check(!registry.unregisterTool("first", "other"), "Provider removed foreign tool");
    bool invalid = false;
    try { registry.registerTool(descriptor("missing_handler"), {}); }
    catch (const std::invalid_argument&) { invalid = true; }
    check(invalid && !registry.isRegistered("missing_handler"), "Metadata-only tool accepted");

    first.description = "Updated description";
    first.aliases = {"new_alias"};
    check(registry.updateMetadata(first), "Metadata update failed");
    check(!registry.isRegistered("alias") && registry.isRegistered("new_alias"), "Metadata update retained stale alias");
    check(registry.getTool("first")->usage_count == 1, "Metadata update reset stats");
    check(dispatchCommand("new_alias", "still callable").message == "still callable", "Metadata update lost handler");
    first.swap_state = ToolSwapState::Unavailable;
    registry.updateMetadata(first);
    check(!registry.isExecutable("first") && registry.getExecutableNames().empty(), "Unavailable tool exposed as executable");
    check(dispatchCommand("first", "").errorCode == "ERR_TOOL_UNAVAILABLE", "Unavailable handler ran");
    first.swap_state = ToolSwapState::Loaded;
    registry.updateMetadata(first);

    auto batchGood = descriptor("batch_good");
    auto batchBad = descriptor("batch_bad");
    invalid = false;
    try { registry.registerBatch("tests", {{batchGood, echo}, {batchBad, {}}}); }
    catch (const std::invalid_argument&) { invalid = true; }
    check(invalid && !registry.isRegistered("batch_good"), "Invalid batch partially committed");
    check(registry.registerBatch("tests", {{batchGood, echo}, {batchBad, echo}}), "Valid batch failed");
    check(registry.getExecutableNames().size() == 3, "Callable listing lost batch tools");
    auto throwing = descriptor("throwing");
    registry.registerTool(throwing, [](const std::string&) -> CommandResult { throw std::runtime_error("test"); });
    check(dispatchCommand("throwing", "").errorCode == "ERR_CMD_EXCEPTION", "Handler exception escaped dispatch");
    check(registry.getTool("throwing")->failure_count == 1, "Failure counted incorrectly");
    auto nested = descriptor("nested");
    registry.registerTool(nested, [](const std::string& args) { return dispatchCommand("first", args); });
    check(dispatchCommand("nested", "nested result").message == "nested result", "Nested registry dispatch failed");

    auto replaced = descriptor("replaced");
    registry.registerTool(replaced, [&](const std::string&) {
        registry.unregisterTool("replaced");
        registry.registerTool(replaced, echo);
        return CommandResult{};
    });
    dispatchCommand("replaced", "");
    check(registry.getTool("replaced")->usage_count == 0, "Old execution updated replacement's stats");
    registry.unregisterByProvider("tests");
    check(registry.toolCount() == 0 && !registry.isRegistered("new_alias"), "Provider cleanup left execution or aliases");
    check(dispatchCommand("new_alias", "").errorCode == "ERR_UNKNOWN_CMD", "Removed handler remained executable");

    grim_register_command("legacy_host", echo);
    check(registry.getTool("legacy_host").has_value(), "Host adapter created a second store");
    check(dispatchCommand("legacy_host", "host").message == "host", "Host adapter handler not in registry");
    grim_unregister_command("legacy_host");
    check(!registry.isRegistered("legacy_host"), "Host adapter removal retained registry tool");
    check(observer.changes > 0, "No registry notifications");
    registry.removeListener(&observer);
    // Classification controls model discovery, not explicit host invocation.
    check(ToolDescriptor{}.usage_type == ToolUsage::Model, "Default usage changed");
    ToolGapPlanner planner(registry);
    for (auto usage : {ToolUsage::Debug, ToolUsage::Model, ToolUsage::System, ToolUsage::Ignore}) {
        auto tool = descriptor(std::string("usage_") + toolUsageName(usage));
        tool.usage_type = usage;
        tool.aliases = {tool.tool_id + "_alias"};
        tool.capability_tags = {tool.tool_id + "_capability"};
        tool.keywords = {tool.tool_id};
        registry.registerTool(tool, echo);
        check(registry.invoke(tool.tool_id, "host").message == "host", "Usage blocked host execution");
        const auto json = nlohmann::json(tool);
        check(json.get<ToolDescriptor>().usage_type == usage, "Usage JSON roundtrip failed");
        auto legacy = json; legacy.erase("usage_type");
        check(legacy.get<ToolDescriptor>().usage_type == ToolUsage::Model, "Legacy snapshot default changed");
        for (const auto& value : nlohmann::json::array({99, -1, 1.5, "Model", 4294967297ULL})) {
            auto bad = json; bad["usage_type"] = value;
            bool rejected = false;
            try { bad.get<ToolDescriptor>(); } catch (const std::exception&) { rejected = true; }
            check(rejected, "Invalid snapshot usage accepted");
        }
        const bool model = usage == ToolUsage::Model;
        check((registry.generateCompactPrompt().find(tool.tool_id) != std::string::npos) == model &&
              (registry.generateFullPrompt().find(tool.tool_id) != std::string::npos) == model,
              "Non-model tool leaked into prompts");
        check(planner.evaluate(tool.tool_id, "usage-test").has_value() != model &&
              planner.evaluate(tool.aliases[0], "usage-test").has_value() != model &&
              planner.evaluate(tool.capability_tags[0], "usage-test").has_value() != model,
              "Capability discovery ignored usage");
    }
    check(registry.getAllTools().size() == 4 && registry.getModelTools().size() == 1,
          "Model filtering lost developer metadata");
    auto model = *registry.getTool("usage_Model");
    model.swap_state = ToolSwapState::Unavailable;
    check(registry.updateMetadata(model) && registry.getModelTools().empty() &&
          planner.evaluate(model.aliases[0], "usage-test").has_value() &&
          planner.evaluate(model.capability_tags[0], "usage-test").has_value(), "Unavailable Model tool remained discoverable");
    model.swap_state = ToolSwapState::Loaded;
    model.usage_type = ToolUsage::System;
    check(registry.updateMetadata(model) && registry.getModelTools().empty() &&
          registry.getTool(model.tool_id)->usage_count == 1, "Reclassification lost stats or model visibility");
    model.usage_type = static_cast<ToolUsage>(99);
    check(!registry.updateMetadata(model), "Invalid usage update accepted");
    invalid = false;
    try { registry.registerTool(model, echo); } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "Invalid usage registration accepted");
    registry.unregisterByProvider("tests");
    std::cout << "Unified tool registry tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
