#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include "console_history.hpp"
#include "ui_config.hpp"

// Draw the full GRIM UI each frame
void drawUI(
    sf::RenderWindow& window,
    sf::Font& font,
    ConsoleHistory& history,
    const std::string& buffer,
    bool caretVisible,
    float& scrollOffsetLines
);
