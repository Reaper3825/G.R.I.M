#include "ui_helpers.hpp"

bool updateCaretBlink(sf::Clock& caretClock, bool caretVisible) {
    if (caretClock.getElapsedTime().asSeconds() > 0.5f) {
        caretVisible = !caretVisible;
        caretClock.restart();
    }
    return caretVisible;
}

void clampScroll(float& scrollOffsetLines, float maxScroll) {
    if (scrollOffsetLines < 0) scrollOffsetLines = 0;
    if (scrollOffsetLines > maxScroll) scrollOffsetLines = maxScroll;
}
