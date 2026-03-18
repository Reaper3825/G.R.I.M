// DynamicSurfacePanel.cpp — renders a UISurfaceSpec as a UIPanel.
//======================================================//

#include "dynamic_surface_panel.hpp"
#include "overlay_renderer.hpp"

using namespace GRIM::MMO;

DynamicSurfacePanel::DynamicSurfacePanel(const UISurfaceSpec& spec)
    : UIPanel(spec.title, /*draggable=*/true)
    , surface_id_(spec.surface_id)
    , spec_(spec)
{
    setResizable(spec.kind == SurfaceKind::ToolWindow ||
                 spec.kind == SurfaceKind::Inspector);

    // Modals block input
    if (spec.kind == SurfaceKind::Modal) {
        setDraggable(false);
    }

    // Toasts are non-interactive chrome-less
    if (spec.kind == SurfaceKind::Toast) {
        setDraggable(false);
        setResizable(false);
        PanelChromeOptions opts;
        opts.enableMinimize = false;
        opts.enableMaximize = false;
        setChromeOptions(opts);
    }

    // Default size
    setSize({320.0f, 200.0f});
    setPosition({100.0f, 100.0f});
}

void DynamicSurfacePanel::updateSpec(const UISurfaceSpec& spec) {
    spec_ = spec;
    setTitle(spec.title);
}

bool DynamicSurfacePanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    Vec2 pos = getPosition();
    float yOff = pos.y + titleBarHeight + static_cast<float>(spec_.layout.padding);
    float xStart = pos.x + static_cast<float>(spec_.layout.padding);
    float spacing = static_cast<float>(spec_.layout.spacing);

    for (const auto& widget : spec_.widgets) {
        if (widget.widget_type == "label") {
            renderer.drawText({xStart, yOff}, widget.label, 0xFFDDDDDD);
            yOff += 20.0f + spacing;
        } else if (widget.widget_type == "button") {
            Vec2 btnSize{100.0f, 24.0f};
            renderer.drawRoundedRect({xStart, yOff}, btnSize, 0xFF444444, 4.0f);
            renderer.drawText({xStart + 8.0f, yOff + 4.0f}, widget.label, 0xFFFFFFFF);
            yOff += btnSize.y + spacing;
        } else if (widget.widget_type == "progress") {
            Vec2 barSize{200.0f, 16.0f};
            renderer.drawRoundedRect({xStart, yOff}, barSize, 0xFF333333, 4.0f);
            renderer.drawRoundedRect({xStart, yOff}, {barSize.x * 0.5f, barSize.y}, 0xFF4488FF, 4.0f);
            yOff += barSize.y + spacing;
        } else {
            renderer.drawText({xStart, yOff}, "[" + widget.widget_type + "] " + widget.label, 0xFF888888);
            yOff += 20.0f + spacing;
        }
    }

    if (spec_.widgets.empty()) {
        renderer.drawText({xStart, yOff}, "(empty surface)", 0xFF666666);
    }
    
    renderer.popClipRect();
    return true;
}
