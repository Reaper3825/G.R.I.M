#include "widget.hpp"
#include "input_parser.hpp"
#include "ui/ui_focus_manager.hpp"

Widget::Widget() = default;

void Widget::update(const InputState&, float) {}
void Widget::draw(UIRenderer&) {}

void Widget::drawOverlay(OverlayRenderer&, const Vec2&) {}

void Widget::setFocused(bool focus) {
    if (focus && !focused) {
        // Gaining focus
        focused = true;
        UIFocusManager::getInstance().setFocusedWidget(focusID, panelID);
    } else if (!focus && focused) {
        // Losing focus
        focused = false;
        if (UIFocusManager::getInstance().getFocusedWidget() == focusID) {
            UIFocusManager::getInstance().clearFocus();
        }
    }
}

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
