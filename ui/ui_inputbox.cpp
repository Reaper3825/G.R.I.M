#include "ui_inputbox.hpp"
#include "input_state.hpp"
#include "ui_renderer.hpp"
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

    // Handle keyboard input (only if focused)
    if (!focused) return;

    for (auto& [code, down] : input.keysDown) {
        if (!down) continue;

        switch (code) {
        case KeyCode::Backspace:
            if (!buffer.empty()) buffer.pop_back();
            break;
        case KeyCode::Enter:
            if (externalBind) *externalBind = buffer;
            buffer.clear();
            break;
        default:
            // ASCII input (rudimentary)
            if ((int)code >= ' ' && (int)code <= '~')
                buffer.push_back(static_cast<char>(code));
            break;
        }
    }
}

void UIInputBox::draw(UIRenderer& renderer) {
    renderer.drawRect(position, size, 0xFF202020);
    std::string display = buffer.empty() ? placeholder : buffer;
    if (caretVisible && focused)
        display.push_back('|');
    renderer.drawText({position.x + 6, position.y + 6}, display, 0xFFFFFFFF);
}
