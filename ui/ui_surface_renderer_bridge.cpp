// UISurfaceRendererBridge.cpp — wires UISurfaceRegistry to UIRoot panels.
//======================================================//

#include "ui_surface_renderer_bridge.hpp"
#include "ui_root.hpp"
#include "dynamic_surface_panel.hpp"
#include "../MMO/UI/UISurfaceRegistry.hpp"

using namespace GRIM::MMO;

void UISurfaceRendererBridge::install() {
    UISurfaceRegistry::instance().onSurfaceChange(
        [](SurfaceEvent event, const std::string& surface_id) {
            switch (event) {
            case SurfaceEvent::Created:
                // Registry callbacks run under its lock. Read the spec only after
                // returning to the UI thread, and use the semantic key for lookup.
                UIRoot::get().postTask([surface_id]() {
                    auto spec = UISurfaceRegistry::instance().get(surface_id);
                    if (!spec) return;
                    auto existing = UIRoot::get().getPanelByKey(surface_id);
                    if (existing) {
                        if (auto* dynamic = dynamic_cast<DynamicSurfacePanel*>(existing.get()))
                            dynamic->updateSpec(*spec);
                        else LOG_ERROR("UI", "Surface key collides with host panel: " + surface_id);
                        return;
                    }
                    auto panel = std::make_shared<DynamicSurfacePanel>(*spec);
                    panel->setVisible(spec->visibility == VisibilityState::Visible);
                    UIRoot::get().addPanel(panel);
                });
                break;
            case SurfaceEvent::Shown:
                UIRoot::get().postTask([surface_id]() {
                    auto panel = UIRoot::get().getPanelByKey(surface_id);
                    if (dynamic_cast<DynamicSurfacePanel*>(panel.get()))
                        UIRoot::get().setVisible(surface_id, true);
                });
                break;

            case SurfaceEvent::Hidden:
                UIRoot::get().postTask([surface_id]() {
                    auto panel = UIRoot::get().getPanelByKey(surface_id);
                    if (dynamic_cast<DynamicSurfacePanel*>(panel.get()))
                        UIRoot::get().setVisible(surface_id, false);
                });
                break;

            case SurfaceEvent::Updated:
                UIRoot::get().postTask([surface_id]() {
                    auto spec = UISurfaceRegistry::instance().get(surface_id);
                    if (!spec) return;
                    auto existing = UIRoot::get().getPanelByKey(surface_id);
                    if (!existing) return;
                    auto* dynamic = dynamic_cast<DynamicSurfacePanel*>(existing.get());
                    if (!dynamic) return;
                    dynamic->updateSpec(*spec);
                    UIRoot::get().setVisible(surface_id, spec->visibility == VisibilityState::Visible);
                });
                break;
            case SurfaceEvent::Destroyed:
                UIRoot::get().postTask([surface_id]() {
                    auto panel = UIRoot::get().getPanelByKey(surface_id);
                    if (dynamic_cast<DynamicSurfacePanel*>(panel.get()))
                        UIRoot::get().removePanel(surface_id);
                });
                break;
            }
        });
}
