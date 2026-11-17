#include "ui_panel_service.hpp"
#include "ui_root.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace GRIM::UI {
namespace {
std::unordered_map<GrimPluginHandle, std::weak_ptr<ProviderPanelLease>> providerLeases;
}
void registerProviderPanelLease(GrimPluginHandle instance, const std::shared_ptr<ProviderPanelLease>& lease) {
    if (!instance || lease->owner.empty() || !providerLeases.emplace(instance, lease).second)
        throw std::logic_error("Invalid or duplicate panel provider instance");
}
void attachNativePanel(UIPanel* panel, GrimPluginHandle instance) {
    auto& root = UIRoot::get();
    root.requireUIThread();
    const auto found = providerLeases.find(instance);
    const auto lease = found == providerLeases.end() ? nullptr : found->second.lock();
    if (!lease || lease->retired || !lease->uiPermitted)
        throw std::logic_error("Native panel attachment requires a live provider with UI permission");
    const auto& key = panel->getPanelKey();
    if (key.empty() || key.size() > 256 || key.find_first_of("\r\n\0", 0, 3) != std::string::npos)
        throw std::invalid_argument("Provider panels require a stable key");
    if (lease->panels.size() >= 64) throw std::logic_error("Provider panel limit exceeded");
    if (root.getPanelByKey(key)) throw std::logic_error("Duplicate panel key: " + key);

    const auto runtime = panel->getPanelID();
    lease->panels.emplace_back(key, runtime);
    try {
        root.registerPanel(panel);
        root.bindPanelResource(key, runtime, lease->owner);
    } catch (...) {
        if (root.getPanelByKey(key) == panel) root.removePanel(key);
        lease->panels.pop_back();
        throw;
    }
}
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease) {
    if (!lease->panels.empty()) UIRoot::get().requireUIThread();
    if (lease->retired.exchange(true)) return;
    std::erase_if(providerLeases, [&](const auto& entry) { return entry.second.lock() == lease; });
    if (lease->panels.empty()) return;
    auto& root = UIRoot::get();
    for (const auto& [key, runtime] : lease->panels) {
        auto* panel = root.getPanelByKey(key);
        if (panel && panel->getPanelID() == runtime) root.removePanel(key);
    }
    lease->panels.clear();
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
