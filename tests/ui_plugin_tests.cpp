#ifndef GRIM_NATIVE_UI_TEST_HOST
#include "plugin_batch_fixture.hpp"
#include "core/plugin_manager.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include "ui/ui_panel_service.hpp"
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::function<void()>> uiTasks;
bool visible = false;
bool completeOnSubmit = false;
int effects = 0;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void pump() { auto tasks = std::move(uiTasks); uiTasks.clear(); for (auto& task : tasks) task(); }
nlohmann::json take(const PluginBatchFixture& provider, const char* value) {
    check(value != nullptr, "UI API returned null unexpectedly");
    const auto release = [&](const char* text) { provider->string_free(provider.handle(), text); };
    std::unique_ptr<const char, decltype(release)> result(value, release);
    return nlohmann::json::parse(result.get());
}
}
namespace GRIM::UI {
// Controlled host boundary: the API/loader/plugin are real, no native UI is started.
// The native provider panel lifecycle is covered by the UI foundation tests.
void registerProviderPanelLease(GrimPluginHandle, const std::shared_ptr<ProviderPanelLease>&) {}
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease) {
    lease->retired = true;
    lease->panels.clear();
}
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key, uint64_t expectedRuntimeId) {
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    auto future = promise->get_future();
    uiTasks.push_back([promise, cancel, operation, key, expectedRuntimeId] {
        if (cancel->load()) { promise->set_value({{"success", false}, {"error", "cancelled"}}); return; }
        if (expectedRuntimeId && expectedRuntimeId != 42) {
            promise->set_value({{"success", false}, {"error", "stale_panel"}}); return;
        }
        if (operation == GRIM_UI_LIST) {
            promise->set_value({{"success", true}, {"panels", {{{"panel_id", "settings"}, {"visible", visible}}}}});
        } else if (key != "settings") {
            promise->set_value({{"success", false}, {"error", "not_found"}});
        } else {
            if (operation == GRIM_UI_SHOW || operation == GRIM_UI_HIDE) { visible = operation == GRIM_UI_SHOW; ++effects; }
            promise->set_value({{"success", true}, {"panel", {{"panel_id", key}, {"visible", visible}}}});
        }
    });
    if (completeOnSubmit) pump();
    return {std::move(future), cancel};
}
}

void runUIPluginTests(const std::filesystem::path& directory) {

    PluginBatchFixture owner({.name = "ui-owner", .path = ""});
    owner.load();
    PluginBatchFixture other({.name = "ui-other", .path = ""});
    other.load();
    check(!owner->ui_panel_service(owner.handle(), GRIM_UI_SHOW, nullptr, 0) &&
          !owner->ui_panel_service(owner.handle(), static_cast<GrimUIPanelOperation>(99), nullptr, 0) &&
          !owner->ui_panel_service(owner.handle(), GRIM_UI_LIST, "unexpected", 0) &&
          !owner->ui_panel_service(owner.handle(), GRIM_UI_SHOW, "settings", 1), "Malformed UI service calls were accepted");
    const auto submitted = take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_SHOW, "settings", 0));
    const auto id = submitted.at("request_id").get<uint64_t>();
    check(submitted.at("status") == "pending" && !visible && effects == 0, "Submission ran UI work synchronously");
    check(take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_POLL, nullptr, id)).at("status") == "pending",
          "Pending request was reported completed");
    check(take(other, other->ui_panel_service(other.handle(), GRIM_UI_POLL, nullptr, id)).at("error") == "request_not_found",
          "Another plugin could read an owned result");
    pump();
    const auto completed = take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_POLL, nullptr, id));
    check(completed.at("result").at("success") && visible && effects == 1, "Completed UI work was not reported");
    check(take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_POLL, nullptr, id)).at("error") == "request_not_found",
          "Completed poll was not consumed");
    const auto cancelled = take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_HIDE, "settings", 0)).at("request_id").get<uint64_t>();
    check(take(other, other->ui_panel_service(other.handle(), GRIM_UI_CANCEL, nullptr, cancelled)).at("error") == "request_not_found",
          "Another provider cancelled an owned request");
    take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_CANCEL, nullptr, cancelled)); pump();
    check(visible && effects == 1, "Cancelled request applied an effect");
    for (int i = 0; i < 64; ++i)
        check(take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_HIDE, "settings", 0)).at("status") == "pending",
              "UI request capacity rejected too early");
    check(take(owner, owner->ui_panel_service(owner.handle(), GRIM_UI_LIST, nullptr, 0)).at("error") == "too_many_requests",
          "UI request capacity was not bounded");
    owner.unload(); pump();
    check(effects == 1 && visible, "Unloaded provider's queued operations still executed");
    other.unload();
    if (directory.empty()) return;
    namespace fs = std::filesystem;
    const auto folder = fs::absolute(directory).parent_path() / "ui-plugin-test-deployment";
    const auto bundle = folder / "ui_plugin";
    fs::create_directories(bundle);
    struct DeploymentCleanup {
        fs::path bundle;
        ~DeploymentCleanup() {
            PluginManager::shutdownBatch();
            std::error_code error;
            fs::remove(bundle / "ui_plugin.dll", error);
            fs::remove(bundle / "ui_plugin.tools.json", error);
            fs::remove(bundle, error);
            fs::remove(bundle.parent_path(), error);
        }
    } cleanup{bundle};
    for (const auto* name : {"ui_plugin.dll", "ui_plugin.tools.json"})
        fs::copy_file(directory / name, bundle / name, fs::copy_options::overwrite_existing);
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    PluginManager::loadBatch(folder.string());
    check(registry.toolCount() == baseline + 6, "UI DLL did not publish six operations");
    for (const char* tool : {"ui.list", "ui.inspect", "ui.show", "ui.hide", "ui.bring_to_front", "ui.poll"}) {
        const auto entry = registry.getTool(tool);
        check(entry && entry->provider.name == "UI Plugin" &&
              entry->argument_format == "json" && !entry->argument_schema_json.empty(), "UI tool metadata incomplete");
    }
    auto invoke = [&registry](const char* tool, const std::string& args) {
        auto result = registry.invoke(tool, args);
        check(result.at("status") == "completed", "UI plugin command failed");
        return result.at("payload");
    };
    const auto count = uiTasks.size();
    for (const char* args : {"", "null", "[]", "{}", "{\"panel_id\":4}", "{\"panel_id\":\"\"}",
                              "{\"panel_id\":\"settings\",\"extra\":true}"})
        check(registry.invoke("ui.show", args).at("error_code") == "ERR_INVALID_ARGS", "Malformed panel arguments accepted");
    for (const char* args : {"{\"request_id\":-1}", "{\"request_id\":0}", "{\"request_id\":1.2}", "{\"request_id\":\"1\"}"})
        check(registry.invoke("ui.poll", args).at("error_code") == "ERR_INVALID_ARGS", "Malformed request ID accepted");
    check(uiTasks.size() == count, "Invalid plugin arguments reached UI service");
    for (const char* args : {"null", "[]", "{\"unexpected\":true}", "garbage"})
        check(registry.invoke("ui.list", args).at("error_code") == "ERR_INVALID_ARGS", "Invalid listing arguments accepted");
    check(uiTasks.size() == count, "Invalid listing reached UI service");
    for (const char* args : {"", " \t "}) {
        const auto bare = invoke("ui.list", args);
        check(bare.at("status") == "pending", "Bare listing did not queue discovery");
        pump();
        const auto complete = invoke("ui.poll", nlohmann::json{{"request_id", bare.at("request_id")}}.dump());
        check(complete.at("result").at("panels")[0].at("panel_id") == "settings",
              "Bare listing did not complete discovery");
    }
    auto listing = invoke("ui.list", "{}");
    pump();
    auto listed = invoke("ui.poll", nlohmann::json{{"request_id", listing.at("request_id")}}.dump());
    check(listed.at("result").at("panels")[0].at("panel_id") == "settings", "UI discovery result was lost");
    completeOnSubmit = true;
    const auto modelList = registry.invoke("ui.list", "{}",
        {.require_canonical_model = true, .wait_for_completion = true});
    completeOnSubmit = false;
    check(modelList.at("status") == "completed" && modelList.at("payload").at("panels")[0].at("panel_id") == "settings" &&
          !modelList.at("payload").contains("request_id"), "UI plugin did not return the completed panel list");
    check(registry.invoke("ui.poll", "{}", {.require_canonical_model = true}).at("error_code") == "ERR_TOOL_NOT_MODEL",
          "UI polling was exposed as a model action");
    const auto effectsBefore = effects;
    auto stale = invoke("ui.show", R"({"panel_id":"settings","expected_runtime_id":41})");
    pump();
    check(registry.invoke("ui.poll", nlohmann::json{{"request_id", stale.at("request_id")}}.dump()).at("status") == "failed" && effects == effectsBefore,
          "Guarded plugin show ignored runtime identity");
    stale = invoke("ui.hide", R"({"panel_id":"settings","expected_runtime_id":41})");
    pump();
    check(registry.invoke("ui.poll", nlohmann::json{{"request_id", stale.at("request_id")}}.dump()).at("status") == "failed" && visible && effects == effectsBefore,
          "Guarded plugin hide ignored runtime identity");
    auto released = invoke("ui.hide", R"({"panel_id":"settings"})");
    invoke("ui.poll", nlohmann::json{{"request_id", released.at("request_id")}, {"cancel", true}}.dump()); pump();
    check(visible && effects == effectsBefore, "Plugin cancellation did not cancel queued work");
    auto hidden = invoke("ui.hide", R"({"panel_id":"settings","expected_runtime_id":42})");
    check(visible, "UI plugin hid panel before processing");
    pump();
    check(invoke("ui.poll", nlohmann::json{{"request_id", hidden.at("request_id")}}.dump()).at("result").at("success") && !visible,
          "UI hide completion was not preserved");
    auto missing = invoke("ui.inspect", "{\"panel_id\":\"missing\"}");
    pump();
    auto failure = registry.invoke("ui.poll", nlohmann::json{{"request_id", missing.at("request_id")}}.dump());
    check(failure.at("status") == "failed" && failure.at("error_code") == "ERR_UI_OPERATION", "Host failure became tool success");
    const auto retired = invoke("ui.show", R"({"panel_id":"settings"})").at("request_id").get<uint64_t>();
    PluginManager::shutdownBatch(); pump();
    check(!visible && registry.toolCount() == baseline, "Unload retained UI operations or queued side effects");

    PluginManager::loadBatch(folder.string());
    const auto current = invoke("ui.list", "{}").at("request_id").get<uint64_t>();
    for (const bool cancel : {false, true}) {
        const auto staleRequest = registry.invoke("ui.poll",
            nlohmann::json{{"request_id", retired}, {"cancel", cancel}}.dump());
        check(staleRequest.at("status") == "failed" && staleRequest.at("error_code") == "ERR_UI_REQUEST" &&
              staleRequest.at("payload").at("error") == "request_not_found",
              "Replacement provider accepted a retired request");
    }
    pump();
    check(invoke("ui.poll", nlohmann::json{{"request_id", current}}.dump()).at("status") == "completed",
          "Retired request disturbed the replacement provider's request");
    PluginManager::shutdownBatch();
    std::cout << "UI plugin metadata, asynchronous completion, ownership and unload tests passed\n";
}

#else
#include <filesystem>
void runUIPluginTests(const std::filesystem::path&) {} // Mock-operation suite runs in the default host; native lifecycle uses the real host.
#endif
