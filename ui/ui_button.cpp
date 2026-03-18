#include "ui_button.hpp"
#include "ui_theme.hpp"
#include "ui_renderer.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"

UIButton::UIButton(const std::string& lbl, std::function<void()> cb)
    : label(lbl), callback(std::move(cb)) {}

void UIButton::update(const InputState& input, float) {
    Vec2 m = input.mousePos;
    
    // Expand hitbox slightly for easier clicking (5px padding)
    float hitPadding = 5.0f;
    bool inside = (m.x >= position.x - hitPadding && m.x <= position.x + size.x + hitPadding &&
                   m.y >= position.y - hitPadding && m.y <= position.y + size.y + hitPadding);
    
    // Track hover state (use exact bounds for visual feedback)
    hovered = (m.x >= position.x && m.x <= position.x + size.x &&
               m.y >= position.y && m.y <= position.y + size.y);
    
    // Use BOTH InputState and Mouse class for maximum reliability
    // This ensures we catch the click even if one source misses it
    bool mouseDown = input.mouseDown[0] || Mouse::isDown(MouseButton::Left);
    bool mousePressed = input.mousePressed[0] || Mouse::wasPressed(MouseButton::Left);
    bool mouseReleased = input.mouseReleased[0] || Mouse::wasReleased(MouseButton::Left);
    
    // Simplified state machine - more forgiving
    if (inside && mousePressed && !pressed) {
        // Mouse pressed inside expanded hitbox - start press
        pressed = true;
        pressedInside = true;
        LOG_DEBUG("UIButton", "Pressed: " + label);
    }
    else if (pressed) {
        // We're in pressed state - waiting for release
        if (mouseReleased || !mouseDown) {
            // Mouse released OR lost mouse down state
            // Trigger callback if we're still near the button (expanded hitbox)
            // This is very forgiving - allows slight mouse movement
            if (inside && callback) {
                LOG_DEBUG("UIButton", "Clicked: " + label);
                callback();
            }
            else {
                LOG_DEBUG("UIButton", "Click cancelled (mouse moved away): " + label);
            }
            pressed = false;
            pressedInside = false;
        }
    }
}

void UIButton::draw(UIRenderer& renderer) {
    uint32_t c = baseColor;
    if (pressed) {
        c = pressColor;
    }
    else if (hovered) {
        c = hoverColor;
    }

    renderer.drawRect(position, size, c);
    renderer.drawText({position.x + 8, position.y + size.y / 4}, label, UITheme::Colors::TextPrimary);
}

void UIButton::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    uint32_t c = baseColor;
    if (pressed) {
        c = pressColor;
    }
    else if (hovered) {
        c = hoverColor;
    }

    renderer.drawRoundedRect(position, size, c, UITheme::Sizes::WidgetRadius);
    
    // Glass border
    renderer.drawRoundedBorder(position, size, UITheme::Colors::BorderPrimary, UITheme::Sizes::WidgetRadius);
    
    // Center text vertically in button
    float textY = position.y + (size.y / 2.0f) - 8;
    renderer.drawText({position.x + 8, textY}, label, UITheme::Colors::TextPrimary);
}
