#include "core/plugin_api_impl.hpp"
#include "core/plugin_manager.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/commands_core.hpp"
#include "ui/ui_panel_service.hpp"
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::function<void()>> uiTasks;
bool visible = false;
int effects = 0;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void pump() { auto tasks = std::move(uiTasks); uiTasks.clear(); for (auto& task : tasks) task(); }
nlohmann::json take(const GrimPluginAPI* api, const char* value) {
    check(value != nullptr, "UI API returned null unexpectedly");
    struct Release { const GrimPluginAPI* api; const char* value; ~Release() { api->string_free(api, value); } } release{api, value};
    return nlohmann::json::parse(value);
}
}
namespace GRIM::UI {
// Controlled host boundary: the API/loader/plugin are real, no native UI is started.
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key) {
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    auto future = promise->get_future();
    uiTasks.push_back([promise, cancel, operation, key] {
        if (cancel->load()) { promise->set_value({{"success", false}, {"error", "cancelled"}}); return; }
        if (operation == GRIM_UI_LIST) {
            promise->set_value({{"success", true}, {"panels", {{{"panel_id", "settings"}, {"visible", visible}}}}});
        } else if (key != "settings") {
            promise->set_value({{"success", false}, {"error", "not_found"}});
        } else {
            if (operation == GRIM_UI_SHOW || operation == GRIM_UI_HIDE) { visible = operation == GRIM_UI_SHOW; ++effects; }
            promise->set_value({{"success", true}, {"panel", {{"panel_id", key}, {"visible", visible}}}});
        }
    });
    return {std::move(future), cancel};
}
}

void runUIPluginTests(const std::filesystem::path& directory) {
    static_assert(sizeof(GrimPluginAPI) == offsetof(GrimPluginAPI, ui_panel_service) + 20 * sizeof(void*));
    auto* denied = PluginAPI::createPluginAPI("ui-denied", "", 0);
    check(!denied->ui_panel_service(denied, GRIM_UI_LIST, nullptr, 0), "UI permission was not enforced");
    PluginAPI::cleanupPluginContext(denied);
    auto* owner = PluginAPI::createPluginAPI("ui-owner", "", GRIM_PERM_UI);
    auto* other = PluginAPI::createPluginAPI("ui-other", "", GRIM_PERM_UI);
    check(!owner->ui_panel_service(owner, GRIM_UI_SHOW, nullptr, 0) &&
          !owner->ui_panel_service(owner, static_cast<GrimUIPanelOperation>(99), nullptr, 0) &&
          !owner->ui_panel_service(owner, GRIM_UI_LIST, "unexpected", 0) &&
          !owner->ui_panel_service(owner, GRIM_UI_SHOW, "settings", 1), "Malformed UI service calls were accepted");
    const auto submitted = take(owner, owner->ui_panel_service(owner, GRIM_UI_SHOW, "settings", 0));
    const auto id = submitted.at("request_id").get<uint64_t>();
    check(submitted.at("status") == "pending" && !visible && effects == 0, "Submission ran UI work synchronously");
    check(take(owner, owner->ui_panel_service(owner, GRIM_UI_POLL, nullptr, id)).at("status") == "pending",
          "Pending request was reported completed");
    check(take(other, other->ui_panel_service(other, GRIM_UI_POLL, nullptr, id)).at("error") == "request_not_found",
          "Another plugin could read an owned result");
    pump();
    const auto completed = take(owner, owner->ui_panel_service(owner, GRIM_UI_POLL, nullptr, id));
    check(completed.at("result").at("success") && visible && effects == 1, "Completed UI work was not reported");
    check(take(owner, owner->ui_panel_service(owner, GRIM_UI_POLL, nullptr, id)).at("error") == "request_not_found",
          "Completed poll was not consumed");
    for (int i = 0; i < 64; ++i)
        check(take(owner, owner->ui_panel_service(owner, GRIM_UI_HIDE, "settings", 0)).at("status") == "pending",
              "UI request capacity rejected too early");
    check(take(owner, owner->ui_panel_service(owner, GRIM_UI_LIST, nullptr, 0)).at("error") == "too_many_requests",
          "UI request capacity was not bounded");
    PluginAPI::cleanupPluginContext(owner); pump();
    check(effects == 1 && visible, "Unloaded provider's queued operations still executed");
    PluginAPI::cleanupPluginContext(other);
    auto* staging = PluginAPI::createPluginAPI("ui-staging", "", GRIM_PERM_UI, true);
    check(!staging->ui_panel_service(staging, GRIM_UI_SHOW, "settings", 0), "Staged initialization submitted UI work");
    PluginAPI::cleanupPluginContext(staging);

    if (directory.empty()) return;
    namespace fs = std::filesystem;
    const auto folder = fs::absolute(directory);
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 6, "UI DLL did not publish six operations");
    for (const char* tool : {"ui.list", "ui.inspect", "ui.show", "ui.hide", "ui.bring_to_front", "ui.poll"}) {
        const auto entry = registry.getTool(tool);
        check(entry && entry->provider_name == "UI Plugin" && entry->permission_bits == GRIM_PERM_UI &&
              entry->argument_format == "json" && !entry->argument_schema_json.empty(), "UI tool metadata incomplete");
    }
    auto invoke = [](const char* tool, const std::string& args) {
        auto result = dispatchCommand(tool, args);
        check(result.success, "UI plugin command failed");
        return nlohmann::json::parse(result.structuredOutputJson);
    };
    const auto count = uiTasks.size();
    for (const char* args : {"", "null", "[]", "{}", "{\"panel_id\":4}", "{\"panel_id\":\"\"}",
                              "{\"panel_id\":\"settings\",\"extra\":true}"})
        check(dispatchCommand("ui.show", args).errorCode == "ERR_INVALID_ARGS", "Malformed panel arguments accepted");
    for (const char* args : {"{\"request_id\":-1}", "{\"request_id\":0}", "{\"request_id\":1.2}", "{\"request_id\":\"1\"}"})
        check(dispatchCommand("ui.poll", args).errorCode == "ERR_INVALID_ARGS", "Malformed request ID accepted");
    check(uiTasks.size() == count, "Invalid plugin arguments reached UI service");
    auto listing = invoke("ui.list", "{}");
    pump();
    auto listed = invoke("ui.poll", nlohmann::json{{"request_id", listing.at("request_id")}}.dump());
    check(listed.at("result").at("panels")[0].at("panel_id") == "settings", "UI discovery result was lost");
    auto hidden = invoke("ui.hide", "{\"panel_id\":\"settings\"}");
    check(visible, "UI plugin hid panel before processing");
    pump();
    check(invoke("ui.poll", nlohmann::json{{"request_id", hidden.at("request_id")}}.dump()).at("result").at("success") && !visible,
          "UI hide completion was not preserved");
    auto missing = invoke("ui.inspect", "{\"panel_id\":\"missing\"}");
    pump();
    auto failure = dispatchCommand("ui.poll", nlohmann::json{{"request_id", missing.at("request_id")}}.dump());
    check(!failure.success && failure.errorCode == "ERR_UI_OPERATION", "Host failure became tool success");
    invoke("ui.show", "{\"panel_id\":\"settings\"}");
    PluginManager::unloadAll(); pump();
    check(!visible && registry.toolCount() == baseline, "Unload retained UI operations or queued side effects");
    std::cout << "UI plugin metadata, permissions, asynchronous completion, ownership and unload tests passed\n";
}
