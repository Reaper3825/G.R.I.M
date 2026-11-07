#include "ui_panel_service.hpp"
#include "ui_root.hpp"
#include <exception>
#include <utility>

namespace GRIM::UI {
void activateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            std::vector<std::string> implementations) {
    UIRoot::get().postTask([lease, implementations = std::move(implementations)] {
        std::lock_guard lock(lease->mutex);
        if (lease->retired) return;
        auto& root = UIRoot::get();
        try {
            lease->panels.reserve(implementations.size());
            for (const auto& implementation : implementations) {
                // Allocate the identity slot before creating a live panel.
                lease->panels.emplace_back();
                auto& identity = lease->panels.back();
                auto panel = root.createPanelFromFactory(implementation);
                try {
                    identity = {panel->getPanelKey(), panel->getPanelID()};
                } catch (...) {
                    root.removePanel(panel->getPanelKey());
                    throw;
                }
                LOG_DEBUG("UIPanelService", lease->owner + " created panel: " + identity.first);
            }
            for (const auto& [key, runtime] : lease->panels)
                root.bindPanelResource(key, runtime, lease->providerName, lease->owner);
        } catch (...) {
            const auto failure = std::current_exception();
            // A provider's panel batch is all-or-nothing on the UI thread.
            for (const auto& [key, runtime] : lease->panels) {
                auto panel = root.getPanelByKey(key);
                if (panel && panel->getPanelID() == runtime) root.removePanel(key);
            }
            lease->panels.clear();
            try { std::rethrow_exception(failure); }
            catch (const std::exception& error) {
                LOG_ERROR("UIPanelService", lease->owner + " panel activation failed: " + error.what());
            } catch (...) {
                LOG_ERROR("UIPanelService", lease->owner + " panel activation failed: unknown exception");
            }
        }
    });
}

void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease) {
    {
        std::lock_guard lock(lease->mutex);
        if (lease->retired) return;
        lease->retired = true;
    }
    UIRoot::get().postTask([lease] {
        std::lock_guard lock(lease->mutex);
        auto& root = UIRoot::get();
        for (const auto& [key, runtime] : lease->panels) {
            auto panel = root.getPanelByKey(key);
            if (panel && panel->getPanelID() == runtime) {
                root.removePanel(key);
                LOG_DEBUG("UIPanelService", lease->owner + " removed panel: " + key);
            }
        }
        lease->panels.clear();
    });
}

namespace {
nlohmann::json metadata(const UIPanelInfo& panel) {
    return {{"panel_id", panel.key}, {"title", panel.title}, {"runtime_id", panel.runtimeId},
            {"visible", panel.visible}, {"minimized", panel.minimized}, {"maximized", panel.maximized},
            {"position", {{"x", panel.position.x}, {"y", panel.position.y}}},
            {"size", {{"width", panel.size.x}, {"height", panel.size.y}}}, {"z_order", panel.zOrder},
            {"operations", {"inspect", "show", "hide", "bring_to_front"}}};
}
}
PanelServiceRequest submitPanelRequest(GrimUIPanelOperation operation, const std::string& key,
                                       uint64_t expectedRuntimeId) {
    auto promise = std::make_shared<std::promise<nlohmann::json>>();
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    auto future = promise->get_future();
    UIRoot::get().postTask([promise, cancelled, operation, key, expectedRuntimeId] {
        try {
            if (cancelled->load()) {
                promise->set_value({{"success", false}, {"error", "cancelled"}});
                return;
            }
            auto& root = UIRoot::get();
            if (expectedRuntimeId) {
                const auto panel = root.getPanelByKey(key);
                if (!panel || panel->getPanelID() != expectedRuntimeId) {
                    promise->set_value({{"success", false}, {"error", "stale_panel"}});
                    return;
                }
            }
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
