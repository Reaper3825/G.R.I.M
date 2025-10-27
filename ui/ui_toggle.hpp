#pragma once
#include "widget.hpp"
#include <functional>
#include <string>

class UIToggle : public Widget {
public:
    UIToggle(const std::string& label, bool initialState, 
             std::function<void(bool)> onChange);

    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override;
    
    void setState(bool state);
    bool getState() const { return enabled; }
    std::string getLabel() const { return label; }
    
    // For overlay rendering
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos);

private:
    std::string label;
    bool enabled;
    std::function<void(bool)> callback;
    
    Vec2 togglePos{0, 0};
    Vec2 toggleSize{50, 25};
};
