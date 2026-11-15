#pragma once
#include "core/host_exports.hpp"
#include "ui/primitives/ui_panel.hpp"
#include <memory>

class UIObservatoryView {
public:
    GRIM_HOST_API UIObservatoryView();
    GRIM_HOST_API ~UIObservatoryView();
    GRIM_HOST_API void update(const InputState&, float dt, const PanelRect&, uint64_t panelId);
    GRIM_HOST_API void draw(OverlayRenderer&, const PanelRect&);
    GRIM_HOST_API void hide();
    GRIM_HOST_API void refreshModels();
    GRIM_HOST_API bool shouldPassThroughAt(float x,float y) const;
    GRIM_HOST_API void collectPassThroughRects(std::vector<PanelRect>&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
