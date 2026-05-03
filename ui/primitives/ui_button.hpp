#pragma once
#include "widget.hpp"
#include "../ui_theme.hpp"
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
    bool pressedInside = false;

    uint32_t baseColor  = UITheme::Colors::WidgetBg;
    uint32_t hoverColor = UITheme::Colors::WidgetBgHover;
    uint32_t pressColor = UITheme::Colors::WidgetBgActive;
};
