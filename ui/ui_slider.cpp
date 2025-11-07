#include "ui_slider.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "logger.hpp"
#include "ui_focus_manager.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

UISlider::UISlider(const std::string& lbl, float minVal, float maxVal, float initialVal,
                   std::function<void(float)> onChange)
    : label(lbl), minValue(minVal), maxValue(maxVal), value(initialVal), callback(std::move(onChange))
{
    // Generate unique focus ID
    focusID = UIFocusManager::getInstance().generateUniqueID();
    
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
    
    // Calculate text box bounds (where value is displayed)
    textBoxPos = {sliderStart.x + sliderSize.x - 50, position.y};
    textBoxSize = {50, 20};
    
    bool overTextBox = (m.x >= textBoxPos.x && m.x <= textBoxPos.x + textBoxSize.x &&
                        m.y >= textBoxPos.y && m.y <= textBoxPos.y + textBoxSize.y);
    
    bool leftDown = Mouse::isDown(MouseButton::Left);
    
    // Handle text editing
    if (editingText) {
        // Handle Backspace and Delete keys
        if (Key::wasPressed(KeyCode::Backspace) || Key::wasPressed(KeyCode::Delete)) {
            if (!textBuffer.empty()) {
                textBuffer.pop_back();
            }
        }
        
        // Handle Escape key - cancel edit
        if (Key::wasPressed(KeyCode::Escape)) {
            value = valueBeforeEdit;
            editingText = false;
            textBuffer.clear();
            UIFocusManager::getInstance().clearFocus();
            return;
        }
        
        // Handle Enter key - commit edit
        if (Key::wasPressed(KeyCode::Enter)) {
            try {
                float newValue = std::stof(textBuffer);
                newValue = std::clamp(newValue, minValue, maxValue);
                if (newValue != value) {
                    value = newValue;
                    if (callback) callback(value);
                }
            } catch (...) {
                // Invalid input - reset to previous value
                value = valueBeforeEdit;
            }
            editingText = false;
            textBuffer.clear();
            UIFocusManager::getInstance().clearFocus();
            return;
        }
        
        // Check for text input (numeric characters only)
        for (char c : input.textInput) {
            if ((c >= '0' && c <= '9') || c == '.' || c == '-') {
                // Allow numbers, decimal point, and negative sign
                textBuffer += c;
            }
            // Ignore other characters (rejects non-numeric input)
        }
        
        // Click outside text box - commit or cancel
        if (leftDown && !overTextBox) {
            try {
                float newValue = std::stof(textBuffer);
                newValue = std::clamp(newValue, minValue, maxValue);
                if (newValue != value) {
                    value = newValue;
                    if (callback) callback(value);
                }
            } catch (...) {
                // Invalid input - reset to previous value
                value = valueBeforeEdit;
            }
            editingText = false;
            textBuffer.clear();
            UIFocusManager::getInstance().clearFocus();
        }
        return; // Don't process slider dragging while editing text
    }
    
    // Start text editing on click
    if (overTextBox && leftDown) {
        editingText = true;
        valueBeforeEdit = value;
        
        // Set focus to this slider
        UIFocusManager::getInstance().setFocusedWidget(focusID, panelID);
        
        // Initialize text buffer with current value
        std::ostringstream oss;
        oss << std::fixed;
        if (maxValue - minValue < 0.01f) {
            oss << std::setprecision(6);
        } else if (maxValue - minValue < 1.0f) {
            oss << std::setprecision(4);
        } else if (maxValue - minValue < 100.0f) {
            oss << std::setprecision(2);
        } else {
            oss << std::setprecision(0);
        }
        oss << value;
        textBuffer = oss.str();
        return;
    }
    
    // Calculate handle bounds (15px wide centered on track)
    float handleX = getHandleX();
    Vec2 handlePos = {handleX - 7.5f, sliderStart.y - 5};
    Vec2 handleSize = {15, 20};
    
    bool overHandle = (m.x >= handlePos.x && m.x <= handlePos.x + handleSize.x &&
                      m.y >= handlePos.y && m.y <= handlePos.y + handleSize.y);
    
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
    // Draw label on the left
    renderer.drawText({position.x, position.y + 10}, label, 0xFFFFFFFF);
    
    // Recalculate slider positions based on current position (for scrolling)
    Vec2 currentSliderStart = {position.x + 150, position.y + 15};
    Vec2 currentSliderSize = {size.x - 160, 10};
    
    // Recalculate text box position
    Vec2 currentTextBoxPos = {currentSliderStart.x + currentSliderSize.x - 50, position.y};
    Vec2 currentTextBoxSize = {50, 20};
    
    // Draw value text box ABOVE the slider bar
    std::string displayText;
    uint32_t textColor;
    uint32_t boxColor;
    
    if (editingText) {
        // Show text buffer with cursor when editing
        displayText = textBuffer + "|";
        textColor = 0xFFFFFF00;  // Yellow when editing
        boxColor = 0xFF00FFFF;   // Cyan border for active editing
    } else {
        // Show formatted value normally
        std::ostringstream oss;
        oss << std::fixed;
        
        // Determine precision based on value range
        if (maxValue - minValue < 0.01f) {
            oss << std::setprecision(6);  // Very small range (e.g., 0.00001 to 0.01)
        } else if (maxValue - minValue < 1.0f) {
            oss << std::setprecision(4);  // Small range (e.g., 0.0 to 1.0)
        } else if (maxValue - minValue < 100.0f) {
            oss << std::setprecision(2);  // Medium range
        } else {
            oss << std::setprecision(0);  // Large range (integers)
        }
        
        oss << value;
        displayText = oss.str();
        textColor = 0xFF00FFFF;  // Cyan normally
        boxColor = 0xFF404040;   // Dark gray border
    }
    
    // Draw text box background and border
    renderer.drawRect(currentTextBoxPos, currentTextBoxSize, 0xFF1A1A1A);
    renderer.drawRect(currentTextBoxPos, {currentTextBoxSize.x, 1}, boxColor);
    renderer.drawRect(currentTextBoxPos, {1, currentTextBoxSize.y}, boxColor);
    renderer.drawRect({currentTextBoxPos.x, currentTextBoxPos.y + currentTextBoxSize.y - 1}, 
                      {currentTextBoxSize.x, 1}, boxColor);
    renderer.drawRect({currentTextBoxPos.x + currentTextBoxSize.x - 1, currentTextBoxPos.y}, 
                      {1, currentTextBoxSize.y}, boxColor);
    
    // Draw text
    renderer.drawText({currentTextBoxPos.x + 3, currentTextBoxPos.y + 2}, displayText, textColor);
    
    // Draw slider track
    renderer.drawRect(currentSliderStart, currentSliderSize, 0xFF404040);
    
    // Draw filled portion
    float fillWidth = currentSliderSize.x * getNormalizedValue();
    renderer.drawRect(currentSliderStart, {fillWidth, currentSliderSize.y}, 0xFF00FFFF);
    
    // Draw handle
    float handleX = currentSliderStart.x + getNormalizedValue() * currentSliderSize.x;
    Vec2 handlePos = {handleX - 7.5f, currentSliderStart.y - 5};
    Vec2 handleSize = {15, 20};
    
    uint32_t handleColor = dragging ? 0xFF00FFFF : 0xFFCCCCCC;
    renderer.drawRect(handlePos, handleSize, handleColor);
    renderer.drawRect(handlePos, {handleSize.x, 2}, 0xFFFFFFFF);
    renderer.drawRect(handlePos, {2, handleSize.y}, 0xFFFFFFFF);
    renderer.drawRect({handlePos.x, handlePos.y + handleSize.y - 2}, {handleSize.x, 2}, 0xFFFFFFFF);
    renderer.drawRect({handlePos.x + handleSize.x - 2, handlePos.y}, {2, handleSize.y}, 0xFFFFFFFF);
}
