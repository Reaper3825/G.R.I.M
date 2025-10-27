#include "ui_button.hpp"
#include "ui_renderer.hpp"
#include "input_parser.hpp"
#include "logger.hpp"  // ? ADD for logging
#include <algorithm>
#ifdef _DEBUG
#include <crtdbg.h>  // ? ADD for heap checking
#endif

UIButton::UIButton(const std::string& lbl, std::function<void()> cb)
    : label(lbl), callback(std::move(cb)) {}

void UIButton::update(const InputState& input, float) {
    Vec2 m = input.mousePos;
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);

    if (inside && input.mousePressed[0]) {
        pressed = true;
    } else if (pressed && input.mouseReleased[0]) {
        pressed = false;
        if (inside && callback) {
            LOG_DEBUG("UIButton", "About to invoke callback for: " + label);
            
            // ? Check heap integrity before callback
            #ifdef _DEBUG
            if (!_CrtCheckMemory()) {
                LOG_ERROR("UIButton", "HEAP CORRUPTED BEFORE CALLBACK!");
            }
            #endif
            
            callback();
            
            LOG_DEBUG("UIButton", "Callback completed for: " + label);
            
            // ? Check heap integrity after callback
            #ifdef _DEBUG
            if (!_CrtCheckMemory()) {
                LOG_ERROR("UIButton", "HEAP CORRUPTED AFTER CALLBACK!");
            }
            #endif
        }
    }
}

void UIButton::draw(UIRenderer& renderer) {
    // Determine if mouse is hovering over button
    bool isHovered = false; // We'll track this in update
    
    uint32_t c = baseColor;
    if (pressed) c = pressColor;
    else if (isHovered) c = hoverColor;

    renderer.drawRect(position, size, c);
    renderer.drawText({position.x + 8, position.y + size.y / 4}, label, 0xFFFFFFFF);
}
