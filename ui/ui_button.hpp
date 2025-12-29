#pragma once
#include "widget.hpp"
#include <string>
#include <functional>

class UIButton : public Widget {
public:
    UIButton(const std::string& label, std::function<void()> onClick);

    void update(const InputState& input, float dt) override;
    void draw(UIRenderer& renderer) override;
    void drawOverlay(class OverlayRenderer& renderer, const Vec2& panelPos) override;

    // Accessors for state
    bool isHovered() const { return hovered; }
    bool isPressed() const { return pressed; }
    
    // Mutators
    void setText(const std::string& text) { label = text; }
    std::string getText() const { return label; }

private:
    std::string label;
    std::function<void()> callback;
    bool pressed = false;
    bool hovered = false;
    bool pressedInside = false;  // ? NEW: Track if press started inside

    uint32_t baseColor = 0xFF303030;
    uint32_t hoverColor = 0xFF404040;
    uint32_t pressColor = 0xFF505050;
};
