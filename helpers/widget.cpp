#include "widget.hpp"
#include "input_parser.hpp"

Widget::Widget() = default;

void Widget::update(const InputState&, float) {}
void Widget::draw(UIRenderer&) {}

void Widget::drawOverlay(OverlayRenderer&, const Vec2&) {}

void Widget::setPosition(float x, float y)
{
    position.x = x;
    position.y = y;
}

void Widget::setSize(float w, float h)
{
    size.x = w;
    size.y = h;
}
