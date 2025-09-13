#include "ui_helpers.hpp"

// Defined in main.cpp
extern sf::Text g_ui_textbox;
extern std::string g_inputBuffer;

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
    g_inputBuffer = text;               // update raw buffer
    g_ui_textbox.setString(g_inputBuffer); // sync with render object
}
