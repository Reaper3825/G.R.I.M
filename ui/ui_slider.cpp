#include "ui_slider.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"  // ? ADD: For debugging
#include <algorithm>
#include <sstream>
#include <iomanip>

UISlider::UISlider(const std::string& lbl, float minVal, float maxVal, float initialVal,
                   std::function<void(float)> onChange)
    : label(lbl), minValue(minVal), maxValue(maxVal), value(initialVal), callback(std::move(onChange))
{
    // Defensive check: ensure label was copied correctly
    if (label.empty()) {
        LOG_ERROR("UISlider", "WARNING: UISlider constructed with empty label");
    } else {
        LOG_DEBUG("UISlider", "Constructed: " + label);
    }
}

void UISlider::setValue(float val) {
    value = std::clamp(val, minValue, maxValue);
}

float UISlider::getNormalizedValue() const {
    if (maxValue == minValue) return 0.0f;
    return (value - minValue) / (maxValue - minValue);
}

float UISlider::getHandleX() const {
    return sliderStart.x + getNormalizedValue() * sliderSize.x;
}

void UISlider::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    
    // Calculate slider bar bounds
    sliderStart = {position.x + 150, position.y + 15};
    sliderSize = {size.x - 160, 10};
    
    // Calculate handle bounds (15px wide centered on track)
    float handleX = getHandleX();
    Vec2 handlePos = {handleX - 7.5f, sliderStart.y - 5};
    Vec2 handleSize = {15, 20};
    
    bool overHandle = (m.x >= handlePos.x && m.x <= handlePos.x + handleSize.x &&
                      m.y >= handlePos.y && m.y <= handlePos.y + handleSize.y);
    
    bool leftDown = Mouse::isDown(MouseButton::Left);
    
    if (overHandle && leftDown && !dragging) {
        dragging = true;
    }
    
    if (dragging) {
        if (leftDown) {
            // Calculate new value from mouse position
            float normalized = (m.x - sliderStart.x) / sliderSize.x;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            
            float newValue = minValue + normalized * (maxValue - minValue);
            
            if (newValue != value) {
                value = newValue;
                if (callback) callback(value);
            }
        } else {
            dragging = false;
        }
    }
}

void UISlider::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UISlider::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw label
    renderer.drawText({position.x, position.y + 10}, label, 0xFFFFFFFF);
    
    // Draw value text
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value;
    renderer.drawText({position.x + size.x - 50, position.y + 10}, oss.str(), 0xFF00FFFF);
    
    // Draw slider track
    renderer.drawRect(sliderStart, sliderSize, 0xFF404040);
    
    // Draw filled portion
    float fillWidth = sliderSize.x * getNormalizedValue();
    renderer.drawRect(sliderStart, {fillWidth, sliderSize.y}, 0xFF00FFFF);
    
    // Draw handle
    float handleX = getHandleX();
    Vec2 handlePos = {handleX - 7.5f, sliderStart.y - 5};
    Vec2 handleSize = {15, 20};
    
    uint32_t handleColor = dragging ? 0xFF00FFFF : 0xFFCCCCCC;
    renderer.drawRect(handlePos, handleSize, handleColor);
    renderer.drawRect(handlePos, {handleSize.x, 2}, 0xFFFFFFFF);
    renderer.drawRect(handlePos, {2, handleSize.y}, 0xFFFFFFFF);
    renderer.drawRect({handlePos.x, handlePos.y + handleSize.y - 2}, {handleSize.x, 2}, 0xFFFFFFFF);
    renderer.drawRect({handlePos.x + handleSize.x - 2, handlePos.y}, {2, handleSize.y}, 0xFFFFFFFF);
}
