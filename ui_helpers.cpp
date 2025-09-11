#include "ui_helpers.hpp"
#include <string>

// External buffer reference (defined in main.cpp)
extern std::string g_ui_textbox;

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

// --- Voice stream helper ---
void ui_set_textbox(const std::string& text) {
    g_ui_textbox = text; // overwrite current input buffer
}
