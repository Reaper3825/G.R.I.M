#include "ui_slider.hpp"
#include "../ui_theme.hpp"
#include "../overlay_renderer.hpp"
#include "../../core/input_parser.hpp"
#include "helpers/mouse.hpp"
#include "helpers/key.hpp"
#include "logger.hpp"
#include "../ui_focus_manager.hpp"
#include <algorithm>
#include <sstream>
#include <iomanip>

UISlider::UISlider(const std::string& lbl, float minVal, float maxVal, float initialVal,
                   std::function<void(float)> onChange, float step)
    : label(lbl), minValue(minVal), maxValue(maxVal), value(initialVal), callback(std::move(onChange)), stepSize(step)
{
    // Generate unique focus ID
    focusID = UIFocusManager::getInstance().generateUniqueID();
    
    // Defensive check: ensure label was copied correctly
    if (label.empty()) {
        LOG_ERROR("UISlider", "WARNING: UISlider constructed with empty label");
    }
}

// Strip trailing zeros after the decimal point from a fixed-precision string.
static std::string trimTrailingZeros(std::string s) {
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (s.back() == '.') s.pop_back();
    }
    return s;
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
    
    // Slider bar = the track itself (no separate box); bar height fits text
    const float barHeight = 24.0f;
    sliderStart = {position.x + 150, position.y + (size.y - barHeight) * 0.5f};
    sliderSize = {size.x - 160, barHeight};
    
    // Entire track is the editable "box" - click anywhere on track to type value
    bool overTrack = (m.x >= sliderStart.x && m.x <= sliderStart.x + sliderSize.x &&
                      m.y >= sliderStart.y && m.y <= sliderStart.y + sliderSize.y);
    
    bool leftDown = Mouse::isDown(MouseButton::Left);
    
    // Handle text editing
    if (editingText) {
        // Backspace / Delete with key repeat
        bool bsFire = false;
        if (Key::wasPressed(KeyCode::Backspace) || Key::wasPressed(KeyCode::Delete)) {
            bsHeldTimer = 0.0f;
            bsRepeatFiring = false;
            bsFire = true;
        } else if (Key::isDown(KeyCode::Backspace) || Key::isDown(KeyCode::Delete)) {
            bsHeldTimer += dt;
            if (!bsRepeatFiring && bsHeldTimer >= kRepeatDelay) {
                bsRepeatFiring = true;
                bsHeldTimer = 0.0f;
                bsFire = true;
            } else if (bsRepeatFiring && bsHeldTimer >= kRepeatRate) {
                bsHeldTimer = 0.0f;
                bsFire = true;
            }
        } else {
            bsHeldTimer = 0.0f;
            bsRepeatFiring = false;
        }
        if (bsFire && !textBuffer.empty()) {
            textBuffer.pop_back();
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
        
        // Paste support (numeric characters only)
        if (input.pasteRequested && !input.pastedText.empty()) {
            for (char c : input.pastedText) {
                if ((c >= '0' && c <= '9') || c == '.' || c == '-')
                    textBuffer += c;
            }
        }

        // Text input (numeric characters only)
        for (char c : input.textInput) {
            if ((c >= '0' && c <= '9') || c == '.' || c == '-') {
                textBuffer += c;
            }
        }
        
        // Click outside track - commit or cancel
        if (leftDown && !overTrack) {
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
    
    // Handle centered on the bar (full bar width for range)
    float handleX = sliderStart.x + sliderSize.x * getNormalizedValue();
    Vec2 handlePos = {handleX - 7.5f, sliderStart.y + (sliderSize.y - 20) * 0.5f};
    Vec2 handleSize = {15, 20};
    bool overHandle = (m.x >= handlePos.x && m.x <= handlePos.x + handleSize.x &&
                       m.y >= handlePos.y && m.y <= handlePos.y + handleSize.y);

    // Start text editing on click (anywhere on track except handle)
    if (overTrack && !overHandle && leftDown) {
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
    
    if (overHandle && leftDown && !dragging) {
        dragging = true;
    }
    
    if (dragging) {
        if (leftDown) {
            // Calculate new value from mouse position (full bar width)
            float normalized = (m.x - sliderStart.x) / sliderSize.x;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            
            float newValue = minValue + normalized * (maxValue - minValue);
            if (stepSize > 0.0f)
                newValue = minValue + std::round((newValue - minValue) / stepSize) * stepSize;
            newValue = std::clamp(newValue, minValue, maxValue);
            
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
    using namespace UITheme;
    
    // Draw label on the left
    renderer.drawText({position.x, position.y + (size.y - 14) * 0.5f}, label, Colors::TextPrimary);
    
    // Slider bar = the only element (track + fill + value + handle, no separate box)
    const float barHeight = 24.0f;
    Vec2 barPos = {position.x + 150, position.y + (size.y - barHeight) * 0.5f};
    Vec2 barSize = {size.x - 160, barHeight};
    float barRadius = barSize.y * 0.5f;
    
    // Bar background (the track - this IS the "box" functionally)
    uint32_t barBg = editingText ? Colors::ContentAreaBg : Colors::SliderTrack;
    uint32_t barBorder = editingText ? Colors::BorderFocus : Colors::BorderPrimary;
    renderer.drawRoundedRect(barPos, barSize, barBg, barRadius);
    renderer.drawRoundedBorder(barPos, barSize, barBorder, barRadius);
    
    // Filled portion (violet) - left part of the bar
    float fillWidth = barSize.x * getNormalizedValue();
    if (fillWidth > 0 && !editingText)
        renderer.drawRoundedRect(barPos, {fillWidth, barSize.y}, Colors::SliderFill, barRadius);
    
    // Value text drawn ON the bar (right-aligned so it stays visible)
    std::string displayText;
    uint32_t textColor;
    if (editingText) {
        displayText = textBuffer;
        textColor = Colors::Warning;
    } else {
        std::ostringstream oss;
        oss << std::fixed;
        if (maxValue - minValue < 0.01f) oss << std::setprecision(6);
        else if (maxValue - minValue < 1.0f) oss << std::setprecision(4);
        else if (maxValue - minValue < 100.0f) oss << std::setprecision(2);
        else oss << std::setprecision(0);
        oss << value;
        displayText = trimTrailingZeros(oss.str());
        textColor = Colors::TextValue;
    }
    float textY = barPos.y + (barSize.y - 14) * 0.5f;
    float textX = barPos.x + barSize.x - 52;  // Right side of bar, room for value
    renderer.drawText({textX, textY}, displayText, textColor);

    // Draw caret as a separate vertical line (avoids font xoff shift from '|' glyph)
    if (editingText) {
        float caretX = textX + renderer.measureTextWidth(displayText);
        renderer.drawRect({caretX, barPos.y + 4.0f}, {1.5f, barSize.y - 8.0f}, Colors::Warning);
    }
    
    // Handle on the bar (only when not editing)
    if (!editingText) {
        float handleX = barPos.x + barSize.x * getNormalizedValue();
        Vec2 handlePos = {handleX - 7.5f, barPos.y + (barSize.y - 20) * 0.5f};
        Vec2 handleSize = {15, 20};
        uint32_t handleColor = dragging ? Colors::SliderHandleActive : Colors::SliderHandle;
        renderer.drawRoundedRect(handlePos, handleSize, handleColor, Sizes::SmallRadius);
        renderer.drawRoundedBorder(handlePos, handleSize, Colors::BorderPrimary, Sizes::SmallRadius);
    }
}
