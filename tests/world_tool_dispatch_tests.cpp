#include "ui/world_lab/world_tool_dispatch.hpp"
#include <iostream>

using Json = nlohmann::json;
using namespace GRIM::MMO;
using GRIM::WorldLab::WorldToolDispatch;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
ToolDescriptor tool(const std::string& id) {
    ToolDescriptor d;
    d.tool_id = d.display_name = id; d.provider_name = "fixture";
    d.provider_instance = "generation-1"; d.argument_format = "json";
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
    {
        // One shared routing boundary covers host and Model invocations.
        auto definition = tool("execution_mode_fixture");
        int calls = 0;
        registry.registerTool(definition, [&](const std::string& args) {
            ++calls;
            CommandResult result;
            result.success = args != "fail";
            result.message = args;
            result.errorCode = result.success ? "ERR_NONE" : "FIXTURE_FAILURE";
            result.structuredOutputJson = Json{{"actual", args}}.dump();
            return result;
        });
        const auto version = registry.version();
        const auto implicit = registry.invoke(definition.tool_id, "implicit");
        check(implicit.success && implicit.message == "implicit" &&
              Json::parse(implicit.structuredOutputJson).at("actual") == "implicit" && calls == 1,
              "Default invocation did not return the real Production result");
        check(registry.invoke(definition.tool_id, "explicit", ToolExecutionMode::Production).success && calls == 2,
              "Explicit Production did not invoke the handler");
        check(registry.invokeModel(definition.tool_id, "model", version).success && calls == 3,
              "Model invocation did not default to Production");
        check(registry.invokeModel(definition.tool_id, "model", version, ToolExecutionMode::Production).success && calls == 4,
              "Explicit Model Production did not invoke the handler");
        const auto failed = registry.invoke(definition.tool_id, "fail");
        check(!failed.success && failed.errorCode == "FIXTURE_FAILURE" && calls == 5,
              "Production failure result was replaced");
        const auto stats = *registry.getTool(definition.tool_id);
        const auto sandbox = registry.invoke(definition.tool_id, "sandbox", ToolExecutionMode::Sandbox);
        check(!sandbox.success && sandbox.errorCode == "ERR_TOOL_SANDBOX_NOT_IMPLEMENTED" && calls == 5,
              "Reserved Sandbox invoked Production or claimed success");
        const auto modelSandbox = registry.invokeModel(definition.tool_id, "sandbox", version, ToolExecutionMode::Sandbox);
        check(!modelSandbox.success && modelSandbox.errorCode == "ERR_TOOL_SANDBOX_NOT_IMPLEMENTED" && calls == 5,
              "Reserved Model Sandbox invoked Production or claimed success");
        const auto invalid = registry.invoke(definition.tool_id, "invalid", static_cast<ToolExecutionMode>(255));
        check(!invalid.success && invalid.errorCode == "ERR_TOOL_EXECUTION_MODE" && calls == 5,
              "Invalid mode invoked Production");
        const auto after = *registry.getTool(definition.tool_id);
        check(after.usage_count == stats.usage_count && after.success_count == stats.success_count &&
              after.failure_count == stats.failure_count && registry.version() == version,
              "Unimplemented execution modes changed Production statistics or metadata");
        registry.unregisterTool(definition.tool_id);
    }
    uint64_t next = 0, runtime = 7;
    bool visible = false, duplicate = false, removed = false, replaceBeforeShow = false, replaceBeforeHide = false;
    int shows = 0, hides = 0, lists = 0, cancels = 0;
    std::map<uint64_t, Json> requests;
    for (const auto& id : {"ui.list", "ui.inspect", "ui.show", "ui.hide"}) {
        registry.registerTool(tool(id), [&, id = std::string(id)](const std::string& text) {
            const auto args = Json::parse(text);
            Json result;
            if (id == "ui.list") {
                ++lists;
                result = {{"success", true}, {"panels", Json::array()}};
                if (!removed) result["panels"].push_back({{"panel_id", "settings"}, {"title", "Settings panel"}, {"runtime_id", runtime}});
                if (duplicate) result["panels"].push_back({{"panel_id", "other"}, {"title", "Settings panel"}, {"runtime_id", 99}});
            } else {
                check(args.at("panel_id") == "settings", "Bad target encoding");
                if (id == "ui.show" && replaceBeforeShow) { ++runtime; replaceBeforeShow = false; }
                if (id == "ui.hide" && replaceBeforeHide) { ++runtime; replaceBeforeHide = false; }
                const bool stale = (id == "ui.show" || id == "ui.hide") && args.at("expected_runtime_id") != runtime;
                result = {{"success", !removed && !stale}, {"error", stale ? "stale_panel" : "not_found"}};
                if (!removed && !stale) {
                    if (id == "ui.show") { visible = true; ++shows; }
                    if (id == "ui.hide") { visible = false; ++hides; }
                    result["panel"] = {{"panel_id", "settings"}, {"title", "Settings panel"}, {"runtime_id", runtime},
                        {"visible", visible}, {"minimized", false}, {"maximized", false}};
                }
            }
            requests[++next] = result;
            return response({{"format", "grim-ui-panel-response"}, {"version", 1}, {"request_id", next}, {"status", "pending"}});
        });
    }
    registry.registerTool(tool("ui.poll"), [&](const std::string& text) {
        const auto args = Json::parse(text);
        const auto id = args.at("request_id").get<uint64_t>();
        auto value = requests.at(id); requests.erase(id);
        if (args.value("cancel", false)) { ++cancels; value = {{"success", true}}; }
        return response({{"format", "grim-ui-panel-response"}, {"version", 1}, {"request_id", id},
            {"status", "completed"}, {"result", value}});
    });
    WorldToolDispatch dispatch;
    uint64_t action = 0;
    auto run = [&](const std::string& command) { return dispatch.execute(command, ++action, [] { return false; }); };
    check(run("ui.show panel:settings").at("status") == "failed" && !shows, "Undiscovered target executed");
    auto listing = run("ui.list");
    check(listing.at("status") == "completed" && listing.at("resources")[0].at("reference") == "panel:settings", "Directory missing");
    check(!listing.at("resources")[0].contains("runtime_id"), "Host generation leaked");
    auto revision = listing.at("revision");
    check(run("ui.list").at("revision") == revision, "Unchanged directory bumped revision");
    check(run("ui.inspect Settings panel").at("resource").at("visible") == false && !shows, "Inspection caused effects");
    const auto before = lists;
    auto shown = run("ui.show panel:settings");
    check(shown.at("status") == "completed" && shown.at("action_id") == action && visible && shows == 1, "Show did not complete exactly once");
    check(lists == before, "Target dispatch rebuilt directory");
    auto hidden = run("ui.hide Settings panel");
    check(hidden.at("status") == "completed" && hidden.at("resource").at("visible") == false && !visible && hides == 1,
          "Hide did not resolve name and complete exactly once");
    check(hidden.at("feedback") == "Settings panel: hidden.", "Hide feedback did not report actual state");
    check(run("ui.bring_to_front panel:settings").at("status") == "failed", "Unsupported action executed");
    check(run("ui.show panel:missing").at("status") == "failed", "Missing reference executed");
    check(run("ui.list\nui.show panel:settings").at("status") == "failed", "Multiple actions accepted");
    duplicate = true; run("ui.list");
    check(run("ui.show Settings panel").at("status") == "failed" && shows == 1, "Ambiguous name executed");
    check(run("ui.hide Settings panel").at("status") == "failed" && hides == 1, "Ambiguous hide executed");
    duplicate = false; run("ui.list");
    ++runtime;
    check(run("ui.show panel:settings").at("status") == "failed" && shows == 1, "Replaced resource executed");
    check(run("ui.list").at("revision") != revision, "Resource generation did not change directory revision");
    replaceBeforeShow = true;
    check(run("ui.show panel:settings").at("status") == "failed" && shows == 1, "Replacement between inspect and show mutated new resource");
    run("ui.list"); visible = true; replaceBeforeHide = true;
    check(run("ui.hide panel:settings").at("status") == "failed" && visible && hides == 1,
          "Replacement between inspect and hide mutated new resource");
    run("ui.list"); removed = true;
    check(run("ui.show panel:settings").at("status") == "failed" && shows == 1, "Removed resource executed");
    removed = false; run("ui.list");
    auto changed = *registry.getTool("ui.show"); changed.usage_type = ToolUsage::System;
    registry.updateMetadata(changed);
    check(run("ui.show panel:settings").at("status") == "failed" && shows == 1, "Reclassified tool executed");
    check(registry.invokeModel("ui.show", "{}", registry.version()).errorCode == "ERR_TOOL_NOT_MODEL", "Model gate ignored classification");
    check(registry.invokeModel("ui.list", "{}", registry.version() - 1).errorCode == "ERR_TOOL_STALE", "Stale registry accepted");
    auto alias = tool("ui.list"); alias.aliases = {"listing"}; registry.updateMetadata(alias);
    check(registry.invokeModel("listing", "{}", registry.version()).errorCode == "ERR_TOOL_NOT_MODEL", "Alias accepted by canonical gate");
    dispatch.clear();
    check(run("ui.inspect panel:settings").at("status") == "failed", "Cleared session retained resources");
    int cancelledChecks = 0;
    auto cancelled = dispatch.execute("ui.list", ++action, [&] { return ++cancelledChecks > 1; });
    check(cancelled.at("status") == "unknown" && cancels == 1 && requests.empty(), "Cancellation leaked owned request or claimed a known effect outcome");
    // Mapping is explicit: readable usage edits must not become executable syntax.
    changed.usage_type = ToolUsage::Model;
    changed.usage = "Show the observed panel {t}";
    registry.updateMetadata(changed);
    run("ui.list");
    check(run("ui.show Settings panel").at("status") == "completed" && shows == 2,
          "Dispatch inferred its mapping from usage prose");
    WorldToolDispatch otherSession;
    check(otherSession.execute("ui.show panel:settings", 1, [] { return false; }).at("status") == "failed" && shows == 2,
          "Resource discovery leaked across sessions");
    // A valid type label is insufficient when the actual handler parameter differs.
    changed.parameters = {{"other_id", "string", "Different contract", true}};
    registry.updateMetadata(changed);
    run("ui.list");
    check(run("ui.show panel:settings").at("status") == "failed" && shows == 2,
          "Adapter accepted an incompatible handler parameter");
    // Directory refresh validates the candidate before replacing observed resources.
    GRIM::WorldLab::UITools::PanelResources resources;
    const Json panel = {{"panel_id", "settings"}, {"title", "Settings panel"}, {"runtime_id", runtime}};
    resources.refresh({{"panels", Json::array({panel})}}, registry.version());
    const auto directoryRevision = resources.revision();
    bool rejected = false;
    try { resources.refresh({{"panels", Json::array({panel, panel})}}, registry.version()); }
    catch (const std::exception&) { rejected = true; }
    check(rejected && resources.revision() == directoryRevision &&
          resources.resolve("panel:settings", registry.version()).runtime == runtime,
          "Rejected directory refresh replaced valid session resources");
    registry.unregisterByProvider("fixture", ToolProviderType::Builtin);
    std::cout << "World tool discovery, resolution, dispatch, generation and cancellation tests passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
