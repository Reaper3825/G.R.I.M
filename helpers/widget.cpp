#include "widget.hpp"
#include "input_parser.hpp"
#include "ui/ui_focus_manager.hpp"
#include "ui/input/ui_input_router.hpp"

Widget::Widget() : focusID(UIFocusManager::getInstance().generateUniqueID()) {}
Widget::~Widget() { GRIM::UI::InputRouter::get().forget(focusID); }

void Widget::setVisible(bool value) {
    visible = value;
    if (!value) cancelInput();
}
void Widget::cancelInput() { GRIM::UI::InputRouter::get().forget(focusID); focused = false; }

InputState Widget::routeInput(const InputState& source) {
    auto& router = GRIM::UI::InputRouter::get();
    if (router.currentPanel) panelID = router.currentPanel;
    router.touch(focusID, [this] { cancelInput(); });
    focused = router.focus.widget == focusID;
    InputState input = source;
    if (!visible || !router.allowsPointer(focusID) ||
        (!hasPointerCapture() && !hasInputPopup() && !input.pointerInClip())) input.suppressPointer();
    if (!router.allowsWheel(focusID)) input.mouseWheelDelta = input.mouseWheelDeltaX = 0;
    const auto keyboardOwner = router.popup ? router.popup : router.focus;
    if (keyboardOwner.widget != focusID) input.suppressKeyboard();
    return input;
}
bool Widget::capturePointer() { return GRIM::UI::InputRouter::get().claimPointer({focusID, panelID}); }
void Widget::releasePointer() { GRIM::UI::InputRouter::get().releasePointer(focusID); }
bool Widget::hasPointerCapture() const { return GRIM::UI::InputRouter::get().ownsPointer(focusID); }
void Widget::openInputPopup() { GRIM::UI::InputRouter::get().openPopup({focusID, panelID}); }
void Widget::closeInputPopup() { GRIM::UI::InputRouter::get().closePopup(focusID); }
bool Widget::hasInputPopup() const { return GRIM::UI::InputRouter::get().ownsPopup(focusID); }
void Widget::consumeScroll() { GRIM::UI::InputRouter::get().consumeWheel(focusID); }

void Widget::update(const InputState&, float) {}
void Widget::draw(UIRenderer&) {}

void Widget::drawOverlay(OverlayRenderer&, const Vec2&) {}

void Widget::setFocused(bool focus) {
    if (focus) {
        // Gaining focus
        focused = true;
        UIFocusManager::getInstance().setFocusedWidget(focusID, panelID);
    } else {
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
