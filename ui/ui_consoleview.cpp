#include "pch.hpp"
#include "ui_consoleview.hpp"
#include "ui_renderer.hpp"
#include "input_state.hpp"
#include <algorithm>


UIConsoleView::UIConsoleView(ConsoleHistory* h)
    : history(h) {}

void UIConsoleView::update(const InputState& input, float) {
    // Scroll using mouse wheel or keys (optional)
    if (input.keysDown.at(KeyCode::PageUp)) scroll(-3);
    if (input.keysDown.at(KeyCode::PageDown)) scroll(3);
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
        renderer.drawText({position.x + 4, y}, ln.text, ln.color);
        y -= lineHeight;
        if (y < position.y) break;
    }
}
