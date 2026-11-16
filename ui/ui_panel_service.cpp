#include "ui_panel_service.hpp"
#include "ui_root.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace GRIM::UI {
namespace {
std::map<std::string, std::weak_ptr<ProviderPanelLease>, std::less<>> providerLeases;
std::unordered_map<UIPanel*, std::weak_ptr<ProviderPanelLease>> nativeOwners;
}
void registerProviderPanelLease(const std::shared_ptr<ProviderPanelLease>& lease) {
    if (lease->owner.empty() || !providerLeases.emplace(lease->owner, lease).second)
        throw std::logic_error("Invalid or duplicate panel provider instance: " + lease->owner);
}
void stageNativePanel(UIPanel* panel, std::string_view providerInstance) {
    UIRoot::get().requireUIThread();
    const auto found = providerLeases.find(providerInstance);
    const auto owner = found == providerLeases.end() ? nullptr : found->second.lock();
    if (!owner || !owner->acceptingPanels)
        throw std::logic_error("Panel provider instance is not accepting construction: " + std::string(providerInstance));
    auto& lease = *owner;
    const auto& key = panel->getPanelKey();
    try {
        if (!lease.uiPermitted) throw std::logic_error("Native panels require UI permission");
        if (key.empty() || key.size() > 256 || key.find_first_of("\r\n\0", 0, 3) != std::string::npos)
            throw std::invalid_argument("Provider panels require a stable key");
        if (lease.nativePanels.size() >= 64) throw std::logic_error("Provider panel limit exceeded");
        for (const auto* existing : lease.nativePanels)
            if (existing->getPanelKey() == key) throw std::logic_error("Duplicate provider panel key: " + key);
        nativeOwners.emplace(panel, owner);
        try { lease.nativePanels.push_back(panel); }
        catch (...) { nativeOwners.erase(panel); throw; }
    } catch (...) { lease.registrationFailed = true; throw; }
}
void forgetNativePanel(UIPanel* panel) {
    const auto found = nativeOwners.find(panel);
    if (found == nativeOwners.end()) return;
    if (auto lease = found->second.lock())
        std::erase(lease->nativePanels, panel);
    nativeOwners.erase(found);
}
bool validateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            const std::shared_ptr<ProviderPanelLease>& replacing) {
    UIRoot::get().requireUIThread();
    if (lease->registrationFailed || lease->retired) return false;
    for (const auto* panel : lease->nativePanels) {
        const auto* existing = UIRoot::get().getPanelByKey(panel->getPanelKey());
        if (!existing) continue;
        if (!replacing || std::find(replacing->panels.begin(), replacing->panels.end(),
            std::make_pair(existing->getPanelKey(), existing->getPanelID())) == replacing->panels.end()) return false;
    }
    return true;
}
void activateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease) {
    lease->acceptingPanels = false;
    UIRoot::get().postTask([lease] {
        if (lease->retired) return;
        auto& root = UIRoot::get();
        try {
            lease->panels.reserve(lease->nativePanels.size());
            for (auto* panel : lease->nativePanels) {
                lease->panels.emplace_back(panel->getPanelKey(), panel->getPanelID());
                root.registerPanel(panel);
            }
            for (const auto& [key, runtime] : lease->panels)
                root.bindPanelResource(key, runtime, lease->providerName, lease->owner);
        } catch (const std::exception& error) {
            for (const auto& [key, runtime] : lease->panels) {
                auto* panel = root.getPanelByKey(key);
                if (panel && panel->getPanelID() == runtime) root.removePanel(key);
            }
            lease->panels.clear();
            LOG_ERROR("UIPanelService", lease->owner + " panel activation failed: " + error.what());
        }
    });
}
void retireProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease) {
    // Detach synchronously, before plugin shutdown destroys objects or unloads code.
    const bool hasPanels = !lease->nativePanels.empty() || !lease->panels.empty();
    if (hasPanels) UIRoot::get().requireUIThread();
    if (lease->retired.exchange(true)) return;
    const auto registered = providerLeases.find(lease->owner);
    if (registered != providerLeases.end() && registered->second.lock() == lease)
        providerLeases.erase(registered);
    if (!hasPanels) return;
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
