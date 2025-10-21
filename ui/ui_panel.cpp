#include "ui_panel.hpp"
#include "ui_renderer.hpp"
#include "input_state.hpp"
#include <algorithm>

UIPanel::UIPanel(const std::string& t, bool drag)
    : title(t), draggable(drag) {}

void UIPanel::addChild(std::shared_ptr<Widget> w) {
    children.push_back(std::move(w));
}

void UIPanel::removeChild(Widget* w) {
    children.erase(std::remove_if(children.begin(), children.end(),
        [&](const std::shared_ptr<Widget>& ptr) { return ptr.get() == w; }),
        children.end());
}

void UIPanel::update(const InputState& input, float dt) {
    Vec2 m = input.mousePos;
    bool inside = (m.x >= position.x && m.x <= position.x + size.x &&
                   m.y >= position.y && m.y <= position.y + size.y);
    hovered = inside;

    // Handle dragging via title bar
    if (draggable) {
        bool inTitleBar = (m.x >= position.x && m.x <= position.x + size.x &&
                           m.y >= position.y && m.y <= position.y + titleBarHeight);

        if (inTitleBar && input.mousePressed[0]) {
            dragging = true;
            dragOffset = {m.x - position.x, m.y - position.y};
        } else if (input.mouseReleased[0]) {
            dragging = false;
        }

        if (dragging) {
            position.x = m.x - dragOffset.x;
            position.y = m.y - dragOffset.y;
        }
    }

    // Update children
    for (auto& c : children) {
        if (c->isVisible()) {
            c->update(input, dt);
        }
    }
}

void UIPanel::draw(UIRenderer& renderer) {
    // Panel background and border
    renderer.drawRect(position, size, bgColor);

    // Title bar
    renderer.drawRect(position, {size.x, titleBarHeight}, 0xFF303030);
    renderer.drawText({position.x + 8, position.y + 4}, title, 0xFFFFFFFF);

    // Simple border
    renderer.drawRect({position.x, position.y}, {size.x, 1}, borderColor);
    renderer.drawRect({position.x, position.y + size.y - 1}, {size.x, 1}, borderColor);
    renderer.drawRect({position.x, position.y}, {1, size.y}, borderColor);
    renderer.drawRect({position.x + size.x - 1, position.y}, {1, size.y}, borderColor);

    // Draw children
    for (auto& c : children)
        if (c->isVisible())
            c->draw(renderer);
}
