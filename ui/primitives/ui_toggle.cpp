#include "ui_toggle.hpp"
#include "../ui_theme.hpp"
#include "../overlay_renderer.hpp"
#include "../../core/input_parser.hpp"
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
    using namespace UITheme;
    
    // Draw label
    renderer.drawText({position.x, position.y + 15}, label, Colors::TextPrimary);
    
    // Recalculate toggle position based on current position (for scrolling)
    Vec2 currentTogglePos = {position.x + size.x - 60, position.y + 8};
    
    // Draw toggle background — glass pill (fully rounded)
    uint32_t bgColor = enabled ? Colors::ToggleBgOn : Colors::ToggleBgOff;
    float pillRadius = toggleSize.y * 0.5f;
    renderer.drawRoundedRect(currentTogglePos, toggleSize, bgColor, pillRadius);
    
    // Draw toggle border — rounded frosted edge
    uint32_t borderColor = enabled ? Colors::Success : Colors::BorderPrimary;
    renderer.drawRoundedBorder(currentTogglePos, toggleSize, borderColor, pillRadius);
    
    // Draw toggle handle — round glass knob
    float handleX = enabled ? (currentTogglePos.x + toggleSize.x - 22) : (currentTogglePos.x + 2);
    Vec2 handlePos = {handleX, currentTogglePos.y + 2};
    Vec2 handleSize = {20, 20};
    
    renderer.drawRoundedRect(handlePos, handleSize, Colors::ToggleHandle, 10.0f);
    renderer.drawRoundedBorder(handlePos, handleSize, Colors::BorderPrimary, 10.0f);
    
    // Draw state text
    std::string stateText = enabled ? "ON" : "OFF";
    renderer.drawText({currentTogglePos.x + (enabled ? 5.0f : 25.0f), currentTogglePos.y + 7}, stateText, Colors::ToggleText);
}
