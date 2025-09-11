#pragma once
#include <SFML/System/Clock.hpp>
#include <string>

bool updateCaretBlink(sf::Clock& caretClock, bool caretVisible);
void clampScroll(float& scrollOffsetLines, float maxScroll);

// Allow external modules (like voice_stream) to set the active textbox text
void ui_set_textbox(const std::string& text);

// Global textbox buffer (declared in main.cpp, extern here)
extern std::string g_ui_textbox;
