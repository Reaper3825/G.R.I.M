#include "pch.hpp"
#include "ui_consoleview.hpp"
#include "../ui_renderer.hpp"
#include "../../core/input_parser.hpp"
#include <algorithm>
#include "core/grim_platform.h"


UIConsoleView::UIConsoleView(ConsoleHistory* h)
    : history(h) {}

void UIConsoleView::update(const InputState& input, float) {
    // Scroll using keys
    if (input.keysDown[VK_PRIOR]) {
        scroll(-3); // Page Up
    }
    if (input.keysDown[VK_NEXT]) {
        scroll(3); // Page Down
    }
}

void UIConsoleView::scroll(float deltaLines) {
    scrollOffset = std::clamp(scrollOffset + deltaLines, 0.0f, 9999.0f);
}

void UIConsoleView::draw(UIRenderer& renderer) {
    if (!history) return;

    auto lines = history->wrapped();
    float y = position.y + size.y - lineHeight;
    int visibleLines = static_cast<int>(size.y / lineHeight);

    int start = (std::max)(0, (int)lines.size() - visibleLines - (int)scrollOffset);
    for (int i = start; i < (int)lines.size(); ++i) {
        const auto& ln = lines[i];
        float lineX = position.x + 4;
        if (ln.alignment == ConsoleHistory::Alignment::Right) {
            constexpr float kApproximateCharacterWidth = 9.0f;
            lineX = std::max(
                position.x + 4,
                position.x + size.x - 4
                    - static_cast<float>(ln.text.size()) * kApproximateCharacterWidth);
        }
        renderer.drawText({lineX, y}, ln.text, ln.color);
        y -= lineHeight;
        if (y < position.y) break;
    }
}
