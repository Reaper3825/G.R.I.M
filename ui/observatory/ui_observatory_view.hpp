#pragma once
#include "ui/primitives/ui_panel.hpp"
#include <memory>

class UIObservatoryView {
public:
    UIObservatoryView();
    ~UIObservatoryView();
    void update(const InputState&, float dt, const PanelRect&, uint64_t panelId);
    void draw(OverlayRenderer&, const PanelRect&);
    void hide();
    void refreshModels();
    bool shouldPassThroughAt(float x,float y) const;
    void collectPassThroughRects(std::vector<PanelRect>&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
