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
            case SurfaceEvent::Created: {
                auto specOpt = UISurfaceRegistry::instance().get(surface_id);
                if (!specOpt) break;

                auto panel = std::make_shared<DynamicSurfacePanel>(specOpt.value());
                UIRoot::get().postTask([panel]() {
                    UIRoot::get().addPanel(panel);
                });
                break;
            }
            case SurfaceEvent::Shown:
                UIRoot::get().postTask([surface_id]() {
                    UIRoot::get().setVisible(surface_id, true);
                });
                break;

            case SurfaceEvent::Hidden:
                UIRoot::get().postTask([surface_id]() {
                    UIRoot::get().setVisible(surface_id, false);
                });
                break;

            case SurfaceEvent::Updated: {
                auto specOpt = UISurfaceRegistry::instance().get(surface_id);
                if (!specOpt) break;

                UISurfaceSpec spec = specOpt.value();
                UIRoot::get().postTask([surface_id, spec]() {
                    auto existing = UIRoot::get().getPanel(surface_id);
                    if (!existing) return;
                    auto* dyn = dynamic_cast<DynamicSurfacePanel*>(existing.get());
                    if (!dyn) return;
                    dyn->updateSpec(spec);
                });
                break;
            }
            case SurfaceEvent::Destroyed:
                UIRoot::get().postTask([surface_id]() {
                    UIRoot::get().removePanel(surface_id);
                });
                break;
            }
        });
}
