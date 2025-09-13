#pragma once
#include <SFML/Graphics.hpp>
#include <string>

// Global textbox object (rendered)
extern sf::Text g_ui_textbox;

// Global raw input buffer (editable)
extern std::string g_inputBuffer;

bool updateCaretBlink(sf::Clock& caretClock, bool caretVisible);
void clampScroll(float& scrollOffsetLines, float maxScroll);

// --- Voice stream helper ---
void ui_set_textbox(const std::string& text);
