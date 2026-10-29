// Actual host panel management, with native window/renderer effects stubbed by
// ui_input_foundation_tests.cpp. No application or model is started.
#include "ui/ui_root.hpp"
#include "ui/ui_surface_renderer_bridge.hpp"
#include "ui/dynamic_surface_panel.hpp"
#include "ui/ui_panel_service.hpp"
#include "MMO/UI/UISurfaceRegistry.hpp"
#include <chrono>
#include <stdexcept>
#include <thread>

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

void runPanelOperationsTests() {
    auto& root = UIRoot::get();
    auto first = std::make_shared<UIPanel>("Original title", true, "test.first");
    auto second = std::make_shared<UIPanel>("Same title", true, "test.second");
    auto third = std::make_shared<UIPanel>("Same title", true, "test.third");
    auto legacy = std::make_shared<UIPanel>("Legacy unkeyed");
    const auto runtimeId = first->getPanelID();
    root.addPanel(first); root.addPanel(second); root.addPanel(third); root.addPanel(legacy);
    root.addPanel(first); // Re-adding the same object is idempotent.
    first->setTitle("Renamed title");
    check(root.getPanelByKey("test.first") == first, "Title change lost stable-key lookup");
    check(root.getPanel("Renamed title") == first, "Legacy title lookup did not follow rename");
    check(!root.getPanel("Original title") && !root.getPanel("Same title"), "Stale/ambiguous title resolved");
    check(first->getPanelID() == runtimeId, "Semantic identity changed input-routing ID");
    root.addPanel(std::make_shared<UIPanel>("Collision", true, "test.first"));
    check(root.getPanelByKey("test.first") == first, "Duplicate key displaced original panel");
    auto list = root.requestPanelList().get();
    check(list.size() == 3 && list[0].key == "test.first", "Discovery is incomplete, duplicated, or unsorted");
    check(list[0].title == "Renamed title" && list[0].runtimeId == runtimeId, "Discovery metadata is stale");
    check(!root.requestPanelOperation("Renamed title", UIPanelOperation::Show).get().success,
          "Generic operation accepted a display title instead of a stable key");
    check(root.requestPanelOperation("test.first", UIPanelOperation::Hide).get().success && !first->isVisible(),
          "Hide did not complete");
    const auto shown = root.requestPanelOperation("test.first", UIPanelOperation::Show).get();
    check(shown.success && shown.panel->visible && first->getZOrder() == -1, "Show did not bring panel forward");
    check(root.requestPanelOperation("test.second", UIPanelOperation::BringToFront).get().success &&
          second->getZOrder() == -1, "Bring-to-front did not address the stable key");

    std::future<UIPanelOperationResult> pending;
    std::future<std::vector<UIPanelInfo>> pendingList;
    std::thread caller([&] {
        pending = root.requestPanelOperation("test.first", UIPanelOperation::Hide);
        pendingList = root.requestPanelList();
    });
    caller.join();
    check(first->isVisible() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::timeout,
          "Worker request mutated UI or reported completion before UI processing");
    root.processPendingTasks();
    check(pending.get().success && !first->isVisible() && !pendingList.get()[0].visible,
          "Queued operations/discovery did not execute in order");
    bool rejected = false;
    std::thread rejectedPump([&] {
        try { root.processPendingTasks(); }
        catch (const std::logic_error&) { rejected = true; return; }
        rejected = false;
    });
    rejectedPump.join();
    check(rejected, "Worker thread was allowed to process UI tasks");
    check(root.requestPanelOperation("test.first", static_cast<UIPanelOperation>(999)).get().error == "invalid_operation",
          "Unknown operation was accepted");
    std::thread removedTarget([&] { pending = root.requestPanelOperation("test.first", UIPanelOperation::Show); });
    removedTarget.join();
    root.removePanel("test.first");
    root.processPendingTasks();
    check(pending.get().error == "not_found", "Removed target produced stale success");
    auto recreated = std::make_shared<UIPanel>("Recreated", true, "test.first");
    root.addPanel(recreated);
    check(root.getPanelByKey("test.first") == recreated && recreated->getPanelID() != runtimeId,
          "Recreation did not preserve semantic identity independently of runtime identity");

    using namespace GRIM::MMO;
    UISurfaceRendererBridge::install();
    auto& surfaces = UISurfaceRegistry::instance();
    UISurfaceSpec spec;
    spec.surface_id = "test.dynamic"; spec.title = "Different dynamic title";
    check(surfaces.create(spec).empty(), "Dynamic surface creation failed");
    root.processPendingTasks(); // Reentrant registry reads must not deadlock during create.
    auto dynamic = root.getPanelByKey(spec.surface_id);
    check(dynamic && dynamic->getTitle() == spec.title && !dynamic->isVisible(),
          "Dynamic panel lost surface key or initial visibility");
    check(root.requestPanelOperation(spec.surface_id, UIPanelOperation::Show).get().success &&
          surfaces.get(spec.surface_id)->visibility == VisibilityState::Visible,
          "Generic show did not synchronize dynamic registry state");
    root.processPendingTasks();
    spec.title = "Updated dynamic title";
    spec.visibility = VisibilityState::Visible;
    check(surfaces.update(spec).empty(), "Dynamic update failed");
    root.processPendingTasks();
    check(root.getPanelByKey(spec.surface_id) == dynamic && dynamic->getTitle() == spec.title,
          "Dynamic title update changed identity");
    surfaces.hide(spec.surface_id); root.processPendingTasks();
    check(!dynamic->isVisible(), "Surface registry hide did not reach keyed panel");
    surfaces.destroy(spec.surface_id); root.processPendingTasks();
    check(!root.getPanelByKey(spec.surface_id), "Destroyed dynamic panel retained stable key");

    // A conflicting dynamic surface must never hide or destroy a host-owned panel.
    spec.surface_id = "test.first";
    check(surfaces.create(spec).empty(), "Collision fixture failed to register");
    root.processPendingTasks();
    surfaces.hide(spec.surface_id); root.processPendingTasks();
    surfaces.destroy(spec.surface_id); root.processPendingTasks();
    check(root.getPanelByKey("test.first") == recreated && recreated->isVisible(),
          "Dynamic lifecycle touched a host panel on a key collision");

    auto serviceList = GRIM::UI::submitPanelRequest(GRIM_UI_LIST, "");
    check(serviceList.result.wait_for(std::chrono::seconds(0)) == std::future_status::timeout,
          "Plugin service bypassed asynchronous UI processing");
    auto cancelledHide = GRIM::UI::submitPanelRequest(GRIM_UI_HIDE, "test.first");
    cancelledHide.cancelled->store(true);
    auto missingPanel = GRIM::UI::submitPanelRequest(GRIM_UI_INSPECT, "missing.panel");
    root.processPendingTasks();
    const auto servicePanels = serviceList.result.get().at("panels");
    check(servicePanels.size() == 3 && servicePanels[0].at("panel_id") == "test.first" &&
          servicePanels[0].at("operations").size() == 4, "Plugin service metadata was incomplete");
    check(cancelledHide.result.get().at("error") == "cancelled" && recreated->isVisible(),
          "Cancelled plugin request changed a real panel");
    check(missingPanel.result.get().at("error") == "not_found", "Plugin service lost host operation failure");

    root.removePanel("test.first"); root.removePanel("test.second"); root.removePanel("test.third");
    root.removePanel("Legacy unkeyed");
    check(root.requestPanelList().get().empty(), "Panel cleanup retained discoverable entries");
}
