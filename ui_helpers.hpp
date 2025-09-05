#pragma once
#include <SFML/System.hpp>

// Toggle caret visibility every 0.5s
bool updateCaretBlink(sf::Clock& caretClock, bool caretVisible);

// Clamp scroll offset to [0, maxScroll]
void clampScroll(float& scrollOffsetLines, float maxScroll);
