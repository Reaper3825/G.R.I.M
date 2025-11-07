#pragma once
#include "widget.hpp"
#include <vector>
#include <memory>

enum class LayoutDirection {
    Horizontal,
    Vertical
};

class UILayoutBox : public Widget {
public:
    UILayoutBox(LayoutDirection dir, float spacing = 5.0f);
    
    void addWidget(std::shared_ptr<Widget> widget);
    void clearWidgets();
    
    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos) override;
    
    void layout(); // Recalculate positions based on direction
    
    void setSpacing(float s) { spacing = s; layout(); }
    float getSpacing() const { return spacing; }
    
private:
    LayoutDirection direction;
    float spacing;
    std::vector<std::shared_ptr<Widget>> widgets;
};

// Convenience aliases
using UIHBox = UILayoutBox;
using UIVBox = UILayoutBox;
