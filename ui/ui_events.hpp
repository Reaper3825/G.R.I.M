#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include <filesystem>
#include <vector>
#include "nlp/nlp.hpp"
#include "console_history.hpp"
#include "timer.hpp"
#include <nlohmann/json_fwd.hpp>

// -------------------------------------------------------------
// Process all SFML events (SFML 3 style):
// - pollEvent() returns std::optional<sf::Event>
// - sf::Event is a std::variant of event structs
// - Access data with std::get_if<sf::Event::TextEntered>(), etc.
// - Keyboard codes use sf::Keyboard::Scancode instead of sf::Keyboard::Key
//
// Returns false if the window should close.
// -------------------------------------------------------------
bool processEvents(
    sf::RenderWindow& window,
    std::string& buffer,
    std::filesystem::path& currentDir,
    std::vector<Timer>& timers,
    nlohmann::json& longTermMemory, 
    ConsoleHistory& history
);
