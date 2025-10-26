#include "ui_inputbox.hpp"
#include "ui_renderer.hpp"
#include "input_parser.hpp"
#include <Windows.h>


UIInputBox::UIInputBox(std::string* bind)
    : externalBind(bind) {}

void UIInputBox::setPlaceholder(const std::string& t) {
    placeholder = t;
}

void UIInputBox::update(const InputState& input, float) {
    // Caret blink
    uint64_t now = GetTickCount64();
    if (now - lastBlink > 500) {
        caretVisible = !caretVisible;
        lastBlink = now;
    }

    // Check if we're focused (for now, assume always focused when visible)
    bool isFocused = isVisible();
    if (!isFocused) return;

    // Handle keyboard input from key map
    for (const auto& [vk, isPressed] : input.keyPressed)
    {
        if (!isPressed)
            continue;

        switch (vk)
        {
        case VK_BACK: // Backspace
            if (!buffer.empty()) buffer.pop_back();
            break;
        case VK_RETURN: // Enter
            if (externalBind) *externalBind = buffer;
            buffer.clear();
            break;
        default:
            if (vk >= 32 && vk <= 126)
                buffer.push_back(static_cast<char>(vk));
            break;
        }
    }
}

void UIInputBox::draw(UIRenderer& renderer) {
    if (!isVisible()) return;
    
    renderer.drawRect(position, size, 0xFF202020);
    std::string display = buffer.empty() ? placeholder : buffer;
    if (caretVisible && isVisible())
        display.push_back('|');
    renderer.drawText({position.x + 6, position.y + 6}, display, 0xFFFFFFFF);
}
