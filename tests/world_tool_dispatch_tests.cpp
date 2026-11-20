#include "ui/world_lab/world_tool_dispatch.hpp"
#include "MMO/Core/Descriptors/ToolDescriptorJson.hpp"
#include <iostream>

using Json = nlohmann::json;
using namespace GRIM::MMO;
using GRIM::WorldLab::WorldToolDispatch;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
ToolDescriptor tool(const std::string& id) {
    ToolDescriptor d;
    d.tool_id = d.display_name = id;
    d.argument_format = "json";
    if (id == "ui.inspect" || id == "ui.show" || id == "ui.hide") {
        d.usage = id + " {t}"; d.bindings = {{"t", "t"}};
        d.parameters = {{"panel_id", "string", "Panel key", true}};
    }
    return d;
}
CommandResult response(const Json& value) {
    CommandResult result; result.structuredOutputJson = value.dump(); return result;
}
}
int main() try {
    auto& registry = ToolRegistry::instance();
    auto& providers = ProviderRegistry::instance();
    {
        // One shared routing boundary covers host and Model invocations.
        auto definition = tool("execution_mode_fixture");
        definition.aliases = {"execution_mode_alias"};
        int calls = 0;
        const ProviderDescriptor executionOwner{ToolProviderType::Plugin, "Execution fixture", "fixture-1", {}};
        check(providers.publish(executionOwner, {{definition, [&](const std::string& args) {
            ++calls;
            CommandResult result;
            result.success = args != "fail";
            result.message = args;
            result.errorCode = result.success ? "ERR_NONE" : "FIXTURE_FAILURE";
            result.structuredOutputJson = Json{{"actual", args}}.dump();
            return result;
        }}}), "Execution fixture publication failed");
        const auto version = registry.version();
        const auto implicit = registry.invoke("execution_mode_alias", "implicit", {.expected_registry_version = version});
        check(implicit.success && implicit.message == "implicit" &&
              Json::parse(implicit.structuredOutputJson).at("actual") == "implicit" && calls == 1,
              "Revision check alone blocked an alias or changed the Production result");
        check(implicit.structuredOutput == Json{{"actual", "implicit"}},
              "Successful invocation did not attach the parsed provider payload");
        check(registry.invoke("execution_mode_alias", "", {.require_canonical_model = true}).errorCode == "ERR_TOOL_NOT_MODEL",
              "Canonical Model check depended on a registry revision");
        check(registry.invoke(definition.tool_id, "explicit", {.mode = ToolExecutionMode::Production}).success && calls == 2,
              "Explicit Production did not invoke the handler");
        check(registry.invoke(definition.tool_id, "model", {.expected_registry_version = version, .require_canonical_model = true}).success && calls == 3,
              "Model invocation did not default to Production");
        check(registry.invoke(definition.tool_id, "model",
                              {.mode = ToolExecutionMode::Production, .expected_registry_version = version,
                               .require_canonical_model = true}).success && calls == 4,
              "Explicit Model Production did not invoke the handler");
        const auto failed = registry.invoke(definition.tool_id, "fail");
        check(!failed.success && failed.errorCode == "FIXTURE_FAILURE" && calls == 5,
              "Production failure result was replaced");
        const auto stats = *registry.getTool(definition.tool_id);
        const auto sandbox = registry.invoke(definition.tool_id, "sandbox", {.mode = ToolExecutionMode::Sandbox});
        check(!sandbox.success && sandbox.errorCode == "ERR_TOOL_SANDBOX_NOT_IMPLEMENTED" && calls == 5,
              "Reserved Sandbox invoked Production or claimed success");
        check(sandbox.structuredOutput == Json::object(),
              "Registry failure unexpectedly had a provider payload");
        const auto modelSandbox = registry.invoke(definition.tool_id, "sandbox",
            {.mode = ToolExecutionMode::Sandbox, .expected_registry_version = version, .require_canonical_model = true});
        check(!modelSandbox.success && modelSandbox.errorCode == "ERR_TOOL_SANDBOX_NOT_IMPLEMENTED" && calls == 5,
              "Reserved Model Sandbox invoked Production or claimed success");
        const auto invalid = registry.invoke(definition.tool_id, "invalid", {.mode = static_cast<ToolExecutionMode>(255)});
        check(!invalid.success && invalid.errorCode == "ERR_TOOL_EXECUTION_MODE" && calls == 5,
              "Invalid mode invoked Production");
        const auto after = *registry.getTool(definition.tool_id);
        check(after.usage_count == stats.usage_count && after.success_count == stats.success_count &&
              after.failure_count == stats.failure_count && registry.version() == version,
              "Unimplemented execution modes changed Production statistics or metadata");
        providers.retire(executionOwner.name, executionOwner.instance);
    }
    {
        ProviderDescriptor provider{ToolProviderType::Plugin, "fixture", "generation-1", "fixture.dll"};
        auto retained = tool("retained_tool");
        auto retired = tool("retired_tool");
        retired.aliases = {"retired_alias"};
        const auto handler = [](const std::string&) { return response(Json::object()); };
        check(!registry.isExecutable(retained.tool_id), "Tool exposed before provider publication");
        check(providers.publish(provider, {{retained, handler}, {retired, handler}}) &&
              registry.getModelTools().size() == 2 && registry.isExecutable("retired_alias"),
              "Provider publication did not expose its tools");
        const auto oldRevision = registry.version();
        const auto oldInstance = provider.instance;
        const auto oldTool = *registry.getTool(retained.tool_id);
        providers.retire(provider.name, oldInstance);
        provider.instance = "generation-2";
        check(providers.publish(provider, {{retained, handler}}),
              "Provider replacement failed");
        providers.retire(provider.name, oldInstance);
        check(registry.getModelTools().size() == 1 && registry.isExecutable(retained.tool_id) &&
              !registry.isExecutable("retired_alias") &&
              registry.getTool(retained.tool_id)->provider == provider &&
              oldTool.provider.instance == oldInstance && !registry.updateMetadata(oldTool) &&
              !Json(registry.getModelTools().front()).contains("provider_instance") &&
              registry.invoke(retained.tool_id, "", {.expected_registry_version = oldRevision, .require_canonical_model = true}).errorCode == "ERR_TOOL_STALE",
              "Replacement exposed retired tools or lost the new generation");
        providers.retire(provider.name, provider.instance);
        check(registry.getModelTools().empty() && registry.getExecutableNames().empty() &&
              registry.invoke(retained.tool_id, "").errorCode == "ERR_UNKNOWN_CMD",
              "Provider unload left tools discoverable or callable");
    }
    const ProviderDescriptor owner{ToolProviderType::Plugin, "World fixture", "fixture-1", {}};
    int listings = 0, calls = 0;
    auto rawTool = tool("fixture.echo");
    rawTool.argument_format = "raw_text";
    rawTool.aliases = {"fixture.alias"};
    auto systemTool = tool("fixture.system");
    systemTool.usage_type = ToolUsage::System;
    check(providers.publish(owner, {
        {tool("ui.list"), [&](const std::string& args) {
            check(Json::parse(args) == Json::object(), "Listing did not get empty JSON args");
            ++listings;
            return response({{"panels", {{{"panel_id", "provider-panel"}}}}});
        }},
        {rawTool, [&](const std::string& args) {
            ++calls;
            return response({{"echo", args}});
        }},
        {tool("fixture.empty"), [](const std::string&) {
            CommandResult result;
            result.message = "Hidden presentation text";
            result.voice = "Hidden voice text";
            return result;
        }},
        {systemTool, [&](const std::string&) {
            ++calls;
            return response(Json::object());
        }}
    }), "World fixture publication failed");
    uint64_t revision = 0;
    const auto roster = registry.getModelTools(&revision);
    check(roster.size() == 3, "World registry roster included a system tool");
    uint64_t action = 0;
    auto run = [&](const std::string& command) {
        return WorldToolDispatch::execute(command, ++action, revision);
    };
    const auto list = run("ui.list");
    check(list.at("action_id") == action && list.at("status") == "completed" && listings == 1 &&
          Json::parse(list.at("feedback").get<std::string>()).at("panels").at(0).at("panel_id") == "provider-panel",
          "Listing did not pass the provider payload to feedback");
    check(run("fixture.echo two words").at("feedback") == R"({"echo":"two words"})" && calls == 1,
          "Non-UI tool did not dispatch through the registry");
    check(run("fixture.empty").at("feedback") == "{}", "Missing payload exposed presentation text");
    check(run("fixture.alias").at("status") == "failed" && run("fixture.system").at("status") == "failed" && calls == 1,
          "World dispatched an alias or system tool");
    providers.retire(owner.name, owner.instance);
    check(providers.publish(owner, {{rawTool, [&](const std::string&) {
        ++calls;
        return response(Json::object());
    }}}), "Replacement fixture publication failed");
    check(run("fixture.echo").at("status") == "failed" && calls == 1, "Stale revision invoked replacement");
    providers.retire(owner.name, owner.instance);
    std::cout << "World registry dispatch and provider feedback tests passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
