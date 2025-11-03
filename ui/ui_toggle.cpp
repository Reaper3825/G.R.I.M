#include "ui_toggle.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"  // ? ADD: For debugging

UIToggle::UIToggle(const std::string& lbl, bool initialState,
                   std::function<void(bool)> onChange)
    : label(lbl), enabled(initialState), callback(std::move(onChange))
{
    // Defensive check: ensure label was copied correctly
    if (label.empty()) {
        LOG_ERROR("UIToggle", "WARNING: UIToggle constructed with empty label");
    } else {
        LOG_DEBUG("UIToggle", "Constructed: " + label);
    }
}

void UIToggle::setState(bool state) {
    enabled = state;
}

void UIToggle::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    
    togglePos = {position.x + size.x - 60, position.y + 8};
    
    bool overToggle = (m.x >= togglePos.x && m.x <= togglePos.x + toggleSize.x &&
                      m.y >= togglePos.y && m.y <= togglePos.y + toggleSize.y);
    
    if (overToggle && Mouse::wasPressed(MouseButton::Left)) {
        enabled = !enabled;
        if (callback) callback(enabled);
    }
}

void UIToggle::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UIToggle::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw label
    renderer.drawText({position.x, position.y + 15}, label, 0xFFFFFFFF);
    
    // Recalculate toggle position based on current position (for scrolling)
    Vec2 currentTogglePos = {position.x + size.x - 60, position.y + 8};
    
    // Draw toggle background
    uint32_t bgColor = enabled ? 0xFF2A4A2A : 0xFF4A2A2A;  // Green if ON, red if OFF
    renderer.drawRect(currentTogglePos, toggleSize, bgColor);
    
    // Draw toggle border
    uint32_t borderColor = enabled ? 0xFF00FF00 : 0xFFFF0000;
    renderer.drawRect(currentTogglePos, {toggleSize.x, 2}, borderColor);
    renderer.drawRect(currentTogglePos, {2, toggleSize.y}, borderColor);
    renderer.drawRect({currentTogglePos.x, currentTogglePos.y + toggleSize.y - 2}, {toggleSize.x, 2}, borderColor);
    renderer.drawRect({currentTogglePos.x + toggleSize.x - 2, currentTogglePos.y}, {2, toggleSize.y}, borderColor);
    
    // Draw toggle handle (circle approximated with rectangle)
    float handleX = enabled ? (currentTogglePos.x + toggleSize.x - 22) : (currentTogglePos.x + 2);
    Vec2 handlePos = {handleX, currentTogglePos.y + 2};
    Vec2 handleSize = {20, 20};
    
    renderer.drawRect(handlePos, handleSize, 0xFFFFFFFF);
    
    // Draw state text
    std::string stateText = enabled ? "ON" : "OFF";
    renderer.drawText({currentTogglePos.x + (enabled ? 5.0f : 25.0f), currentTogglePos.y + 7}, stateText, 0xFF000000);
}
