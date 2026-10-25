#pragma once
#include "ui_panel.hpp"

// Geometry-only container: views retain ownership of their content. Horizontal
// divides left/right; Vertical divides top/bottom. Coordinates are absolute.
class UISplitPane : public Widget {
public:
    enum class Axis { Horizontal, Vertical };
    explicit UISplitPane(Axis axis):axis_(axis) {}
    void setMinimumSizes(float first, float second);
    void setDefaultSecondSize(float pixels);
    void setSecondSize(float pixels);
    void reset();
    PanelRect firstRect() const;
    PanelRect secondRect() const;
    PanelRect dividerRect() const;
    bool isResizing() const {return dragging_;}
    bool hasCustomSize() const {return custom_;}
    void update(const InputState&, float) override;
    void drawOverlay(OverlayRenderer&, const Vec2&) override;
    void cancelInput() override;
private:
    float secondExtent() const;
    float extent() const;
    float dividerExtent() const;
    Axis axis_;
    float firstMinimum_=80,secondMinimum_=80,defaultSecond_=200,preferredSecond_=200;
    float startPointer_=0,startExtent_=0,savedPreference_=0;
    bool custom_=false,savedCustom_=false,dragging_=false,hovered_=false;
};
