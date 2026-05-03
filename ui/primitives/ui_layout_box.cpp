#include "ui_layout_box.hpp"

UILayoutBox::UILayoutBox(LayoutDirection dir, float spacing)
    : direction(dir), spacing(spacing) {
}

void UILayoutBox::addWidget(std::shared_ptr<Widget> widget) {
    if (widget) {
        widgets.push_back(widget);
        layout();
    }
}

void UILayoutBox::clearWidgets() {
    widgets.clear();
}

void UILayoutBox::update(const InputState& input, float dt) {
    for (auto& widget : widgets) {
        widget->update(input, dt);
    }
}

void UILayoutBox::draw(UIRenderer& renderer) {
    for (auto& widget : widgets) {
        widget->draw(renderer);
    }
}

void UILayoutBox::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    for (auto& widget : widgets) {
        widget->drawOverlay(renderer, panelPos);
    }
}

void UILayoutBox::layout() {
    if (widgets.empty()) return;
    
    Vec2 currentPos = position;
    
    for (auto& widget : widgets) {
        widget->setPosition(currentPos.x, currentPos.y);
        
        if (direction == LayoutDirection::Horizontal) {
            currentPos.x += widget->getSize().x + spacing;
        } else { // Vertical
            currentPos.y += widget->getSize().y + spacing;
        }
    }
    
    // Update our size to encompass all widgets
    if (direction == LayoutDirection::Horizontal) {
        float totalWidth = 0;
        float maxHeight = 0;
        for (size_t i = 0; i < widgets.size(); ++i) {
            totalWidth += widgets[i]->getSize().x;
            if (i < widgets.size() - 1) totalWidth += spacing;
            maxHeight = std::max(maxHeight, widgets[i]->getSize().y);
        }
        size = {totalWidth, maxHeight};
    } else { // Vertical
        float maxWidth = 0;
        float totalHeight = 0;
        for (size_t i = 0; i < widgets.size(); ++i) {
            maxWidth = std::max(maxWidth, widgets[i]->getSize().x);
            totalHeight += widgets[i]->getSize().y;
            if (i < widgets.size() - 1) totalHeight += spacing;
        }
        size = {maxWidth, totalHeight};
    }
}
