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
        check(sandbox.structuredOutput.is_null(),
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
    uint64_t next = 0, runtime = 7;
    bool visible = false, loseCompletion = false;
    int mutations = 0, inspections = 0, cancels = 0;
    std::map<uint64_t, Json> requests;
    const ProviderDescriptor uiOwner{ToolProviderType::Plugin, "UI fixture", "fixture-1", {}};
    std::vector<ToolRegistration> registrations;
    for (const auto& id : {"ui.list", "ui.inspect", "ui.show", "ui.hide"}) {
        registrations.push_back({tool(id), [&, id = std::string(id)](const std::string& text) {
            const auto args = Json::parse(text);
            check(id != "ui.list", "Object discovery queried the native UI instead of published resources");
            check(args.at("panel_id") == "settings", "Incorrect UI key");
            if (id == "ui.inspect") ++inspections;
            else {
                check(args.at("expected_runtime_id") == runtime, "Mutation omitted the generation guard");
                visible = id == "ui.show";
                ++mutations;
            }
            requests[++next] = {{"success", true}, {"panel", {{"panel_id", "settings"},
                {"title", "Settings"}, {"runtime_id", runtime}, {"visible", visible}}}};
            return response({{"format", "grim-ui-panel-response"}, {"version", 1},
                {"request_id", next}, {"status", "pending"}});
        }});
    }
    registrations.push_back({tool("ui.poll"), [&](const std::string& text) {
        const auto args = Json::parse(text);
        const auto id = args.at("request_id").get<uint64_t>();
        if (loseCompletion && !args.value("cancel", false))
            return response({{"format", "grim-ui-panel-response"}, {"version", 1},
                {"request_id", id}, {"status", "error"}, {"error", "request_not_found"}});
        const auto value = requests.at(id);
        requests.erase(id);
        if (args.value("cancel", false)) ++cancels;
        return response({{"format", "grim-ui-panel-response"}, {"version", 1}, {"request_id", id},
            {"status", "completed"}, {"result", value}});
    }});
    check(providers.publish(uiOwner, registrations), "UI fixture publication failed");
    ResourceDescriptor panel{"panel:settings", "ui.panel", "Settings", {}, runtime, {{"visible", false}}};
    check(providers.publishResource(uiOwner.instance, panel), "Panel resource publication failed");
    uint64_t action = 0;
    auto run = [&](const std::string& command) {
        return WorldToolDispatch::execute(command, ++action, [] { return false; });
    };
    const auto list = run("ui.list");
    const auto ref = list.at("resources")[0].at("reference").get<std::string>();
    check(list.at("resources")[0].at("kind") == "object", "Native panel leaked into the world");
    check(run("ui.inspect " + ref).at("resource").at("state").at("visible") == false && inspections == 1,
          "Inspection did not return object state");
    const auto shown = run("ui.show " + ref);
    check(shown.at("status") == "completed" && shown.at("action_id") == action &&
          shown.at("resource").at("state").at("visible") == true && mutations == 1 && inspections == 1,
          "Show did not execute exactly once and return object state");
    check(run("ui.bring_to_front " + ref).at("status") == "failed", "Unapproved tool executed");
    providers.removeResource(uiOwner.instance, panel);
    panel.runtime_id = ++runtime;
    check(providers.publishResource(uiOwner.instance, panel), "Replacement resource publication failed");
    check(run("ui.hide " + ref).at("status") == "failed" && mutations == 1, "Stale object changed replacement");
    const auto replacement = run("ui.list").at("resources")[0].at("reference").get<std::string>();
    check(run("ui.hide " + replacement).at("resource").at("state").at("visible") == false && mutations == 2,
          "Replacement object could not be targeted");
    loseCompletion = true;
    check(run("ui.inspect " + replacement).at("status") == "unknown" && cancels == 1 && requests.empty(),
          "Lost completion did not cancel the outstanding request or report uncertainty");
    providers.retire(uiOwner.name, uiOwner.instance);
    std::cout << "World tool discovery, resolution, dispatch, generation and cancellation tests passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
