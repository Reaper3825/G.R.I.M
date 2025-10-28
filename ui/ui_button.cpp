#include "ui_button.hpp"
#include "ui_renderer.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"

UIButton::UIButton(const std::string& lbl, std::function<void()> cb)
    : label(lbl), callback(std::move(cb)) {}

void UIButton::update(const InputState& input, float) {
    Vec2 m = input.mousePos;
    
    // Check if mouse is over button
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);
    
    // Track hover state
    hovered = inside;
    
    // Use Mouse class directly (bypasses InputState filtering for UI)
    bool mouseDown = Mouse::isDown(MouseButton::Left);
    bool mousePressed = Mouse::wasPressed(MouseButton::Left);
    bool mouseReleased = Mouse::wasReleased(MouseButton::Left);
    
    // ? FIX: Check mouse state even if input filtering is disabled
    // The Mouse class tracks the actual hardware state, so we can use it
    // to complete button clicks even when InputState filtering changes
    
    // Button state machine
    if (inside && mousePressed && !pressed) {
        // Mouse pressed inside button
        pressed = true;
        pressedInside = true;  // ? Remember we started inside
        LOG_DEBUG("UIButton", "Pressed: " + label);
    }
    else if (pressed && mouseReleased) {
        // Mouse released - trigger callback if we started inside
        pressed = false;
        
        // ? FIX: Use pressedInside flag instead of current 'inside' state
        // This way the callback can change UI layout without affecting the click
        if (pressedInside && callback) {
            LOG_DEBUG("UIButton", "Clicked: " + label);
            callback();
        }
        
        pressedInside = false;
    }
    else if (pressed && !mouseDown) {
        // Lost mouse down state
        if (!input.mouseInputEnabled) {
            // ? Expected: Input filtering disabled (panel closed)
            // Still trigger the callback if we were pressed inside
            if (pressedInside && callback) {
                LOG_DEBUG("UIButton", "Clicked (filter disabled): " + label);
                callback();
            }
        }
        else {
            // Unexpected loss of state
            LOG_DEBUG("UIButton", "Press state lost: " + label);
        }
        
        pressed = false;
        pressedInside = false;
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
    renderer.drawText({position.x + 8, position.y + size.y / 4}, label, 0xFFFFFFFF);
}

void UIButton::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    uint32_t c = baseColor;
    if (pressed) {
        c = pressColor;
    }
    else if (hovered) {
        c = hoverColor;
    }

    renderer.drawRect(position, size, c);
    
    // Center text vertically in button
    float textY = position.y + (size.y / 2.0f) - 8;
    renderer.drawText({position.x + 8, textY}, label, 0xFFFFFFFF);
}
