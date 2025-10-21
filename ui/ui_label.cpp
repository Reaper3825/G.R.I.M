#include "ui_label.hpp"
#include "ui_renderer.hpp"

UILabel::UILabel(const std::string& t, uint32_t c)
    : text(t), color(c) {}

void UILabel::setText(const std::string& t) { text = t; }
void UILabel::setColor(uint32_t c) { color = c; }

void UILabel::update(const InputState&, float) {
    // No logic needed yet (passive element)
}

void UILabel::draw(UIRenderer& renderer) {
    renderer.drawText(position, text, color);
}
