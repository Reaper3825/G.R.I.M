#include "ui_panel_service.hpp"
#include "ui_root.hpp"

namespace GRIM::UI {
namespace {
nlohmann::json metadata(const UIPanelInfo& panel) {
    return {{"panel_id", panel.key}, {"title", panel.title}, {"runtime_id", panel.runtimeId},
            {"visible", panel.visible}, {"minimized", panel.minimized}, {"maximized", panel.maximized},
            {"position", {{"x", panel.position.x}, {"y", panel.position.y}}},
            {"size", {{"width", panel.size.x}, {"height", panel.size.y}}}, {"z_order", panel.zOrder},
            {"operations", {"inspect", "show", "hide", "bring_to_front"}}};
}
}
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key) {
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    auto future = promise->get_future();
    UIRoot::get().postTask([promise, cancelled, operation, key] {
        try {
            if (cancelled->load()) {
                promise->set_value({{"success", false}, {"error", "cancelled"}});
                return;
            }
            auto& root = UIRoot::get();
            if (operation == GRIM_UI_LIST) {
                auto panels = nlohmann::json::array();
                // The task is now on the UI thread; these host futures are ready immediately.
                for (const auto& panel : root.requestPanelList().get()) panels.push_back(metadata(panel));
                promise->set_value({{"success", true}, {"panels", std::move(panels)}});
                return;
            }
            UIPanelOperation mapped;
            switch (operation) {
            case GRIM_UI_INSPECT: mapped = UIPanelOperation::Inspect; break;
            case GRIM_UI_SHOW: mapped = UIPanelOperation::Show; break;
            case GRIM_UI_HIDE: mapped = UIPanelOperation::Hide; break;
            case GRIM_UI_BRING_TO_FRONT: mapped = UIPanelOperation::BringToFront; break;
            default:
                promise->set_value({{"success", false}, {"error", "invalid_operation"}});
                return;
            }
            const auto result = root.requestPanelOperation(key, mapped).get();
            nlohmann::json value = {{"success", result.success}, {"error", result.error}};
            if (result.panel) value["panel"] = metadata(*result.panel);
            promise->set_value(std::move(value));
        } catch (...) { promise->set_exception(std::current_exception()); }
    });
    return {std::move(future), std::move(cancelled)};
}
}
