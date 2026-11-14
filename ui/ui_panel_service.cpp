#include "ui_panel_service.hpp"
#include "ui_root.hpp"
#include "provider_defined_panel.hpp"
#include "panel_definition.hpp"
#include <exception>
#include <utility>

namespace GRIM::UI {
void activateProviderPanels(const std::shared_ptr<ProviderPanelLease>& lease,
                            std::vector<std::string> implementations,
                            std::vector<nlohmann::json> definitions) {
    UIRoot::get().postTask([lease, implementations = std::move(implementations), definitions = std::move(definitions)] {
        std::lock_guard lock(lease->mutex);
        if (lease->retired) return;
        auto& root = UIRoot::get();
        try {
            lease->panels.reserve(implementations.size() + definitions.size());
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
            for (const auto& definition : definitions) {
                const auto key = definition.at("panel_id").get<std::string>();
                if (root.getPanelByKey(key)) throw std::runtime_error("Panel key already exists: " + key);
                auto panel = std::make_shared<ProviderDefinedPanel>(definition, lease);
                lease->panels.emplace_back(key, panel->getPanelID());
                root.addPanel(panel);
                if (root.getPanelByKey(key) != panel) throw std::runtime_error("Panel attachment failed: " + key);
                lease->definitionRevisions.emplace(key, definition.at("revision").get<uint64_t>());
            }
            for (const auto& [key, runtime] : lease->panels)
                root.bindPanelResource(key, runtime, lease->providerName, lease->owner);
            std::lock_guard eventLock(lease->eventsMutex);
            for (const auto& definition : definitions) {
                const auto key = definition.at("panel_id").get<std::string>();
                lease->events.push_back({{"type", "activated"}, {"panel_id", key},
                    {"runtime_id", root.getPanelByKey(key)->getPanelID()}, {"success", true}});
            }
        } catch (...) {
            const auto failure = std::current_exception();
            // A provider's panel batch is all-or-nothing on the UI thread.
            for (const auto& [key, runtime] : lease->panels) {
                auto panel = root.getPanelByKey(key);
                if (panel && panel->getPanelID() == runtime) root.removePanel(key);
            }
            lease->panels.clear();
            lease->definitionRevisions.clear();
            std::lock_guard eventLock(lease->eventsMutex);
            for (const auto& definition : definitions)
                lease->events.push_back({{"type", "activated"}, {"panel_id", definition.at("panel_id")},
                    {"runtime_id", uint64_t{0}}, {"success", false}, {"error", "Panel activation failed; see UIPanelService log"}});
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
        { std::lock_guard eventLock(lease->eventsMutex); lease->events.clear(); }
        lease->definitionRevisions.clear();
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

void updateProviderPanel(const std::shared_ptr<ProviderPanelLease>& lease, nlohmann::json definition) {
    validatePanelDefinition(definition);
    const auto key = definition.at("panel_id").get<std::string>();
    const auto runtime = definition.at("runtime_id").get<uint64_t>();
    const auto revision = definition.at("revision").get<uint64_t>();
    std::lock_guard lock(lease->mutex);
    if (lease->retired) throw std::runtime_error("Panel owner retired");
    const auto found = lease->definitionRevisions.find(key);
    if (!runtime || found == lease->definitionRevisions.end() || revision <= found->second ||
        std::find(lease->panels.begin(), lease->panels.end(), std::make_pair(key, runtime)) == lease->panels.end())
        throw std::invalid_argument("Panel update has an unknown, stale or foreign identity");
    UIRoot::get().postTask([lease, definition = std::move(definition), key, runtime] {
        std::lock_guard lock(lease->mutex);
        if (lease->retired) return;
        try {
            auto panel = UIRoot::get().getPanelByKey(key);
            if (!panel || panel->getPanelID() != runtime) throw std::runtime_error("Panel instance no longer exists");
            auto* defined = dynamic_cast<ProviderDefinedPanel*>(panel.get());
            if (!defined) throw std::logic_error("Panel is not provider-defined");
            defined->applyDefinition(definition);
        } catch (const std::exception& error) {
            std::lock_guard eventLock(lease->eventsMutex);
            if (lease->events.size() >= 4096) throw std::runtime_error("Provider panel event queue exhausted");
            lease->events.push_back({{"type", "error"}, {"panel_id", key}, {"runtime_id", runtime},
                {"revision", definition.at("revision")}, {"error", error.what()}});
            LOG_ERROR("UIPanelService", error.what());
        }
    });
    found->second = revision;
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
