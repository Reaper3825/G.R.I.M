#include "ui_textarea.hpp"
#include "overlay_renderer.hpp"
#include "input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"
#include <algorithm>

UITextArea::UITextArea(const std::string& lbl, const std::string& initialText,
                       std::function<void(const std::string&)> onChange)
    : label(lbl), text(initialText), callback(std::move(onChange))
{
    placeholder = "Enter custom personality prompt...";
    updateLines();
    
    if (label.empty()) {
        LOG_ERROR("UITextArea", "WARNING: UITextArea constructed with empty label");
    } else {
        LOG_DEBUG("UITextArea", "Constructed: " + label);
    }
}

void UITextArea::setText(const std::string& newText) {
    text = newText;
    updateLines();
    cursorPos = (std::min)(cursorPos, static_cast<int>(text.length()));
}

void UITextArea::updateLines() {
    lines.clear();
    std::string current;
    
    for (char c : text) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    lines.push_back(current);
    
    if (lines.empty()) {
        lines.push_back("");
    }
}

void UITextArea::handleInput(const InputState& input) {
    // For simplicity, we'll just allow basic text display
    // Full text editing would require keyboard input integration
    // This is a display-only widget for now - editing done via external means
}

void UITextArea::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    
    // Check if mouse is over the text area
    hovered = (m.x >= position.x && m.x <= position.x + size.x &&
               m.y >= position.y && m.y <= position.y + size.y);
    
    // Handle focus
    if (hovered && Mouse::wasPressed(MouseButton::Left)) {
        focused = true;
    } else if (!hovered && Mouse::wasPressed(MouseButton::Left)) {
        focused = false;
    }
    
    if (focused) {
        handleInput(input);
    }
}

void UITextArea::draw(UIRenderer& renderer) {
    // Not used - we use drawOverlay instead
}

void UITextArea::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    // Draw label
    renderer.drawText({position.x, position.y - 18}, label, 0xFFFFFFFF);
    
    // Draw text area background
    uint32_t bgColor = focused ? 0xFF3A3A3A : (hovered ? 0xFF2F2F2F : 0xFF252525);
    renderer.drawRect(position, size, bgColor);
    
    // Draw border
    uint32_t borderColor = focused ? 0xFF00AAFF : 0xFF555555;
    renderer.drawRect(position, {size.x, 2}, borderColor);
    renderer.drawRect(position, {2, size.y}, borderColor);
    renderer.drawRect({position.x, position.y + size.y - 2}, {size.x, 2}, borderColor);
    renderer.drawRect({position.x + size.x - 2, position.y}, {2, size.y}, borderColor);
    
    // Draw text content (multi-line)
    float yPos = position.y + 8;
    float lineHeight = 16.0f;
    int maxLines = static_cast<int>((size.y - 16) / lineHeight);
    
    if (text.empty() && !focused) {
        // Show placeholder
        renderer.drawText({position.x + 8, yPos}, placeholder, 0xFF777777);
    } else {
        // Show actual text
        int startLine = static_cast<int>(scrollOffset);
        for (size_t i = startLine; i < lines.size() && i < startLine + maxLines; ++i) {
            std::string displayLine = lines[i];
            
            // Truncate if too long
            if (displayLine.length() > 60) {
                displayLine = displayLine.substr(0, 57) + "...";
            }
            
            renderer.drawText({position.x + 8, yPos}, displayLine, 0xFFCCCCCC);
            yPos += lineHeight;
        }
    }
    
    // Draw character count
    std::string charCount = std::to_string(text.length()) + " chars";
    renderer.drawText({position.x + size.x - 80, position.y + size.y - 18}, charCount, 0xFF888888);
}
