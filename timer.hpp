#pragma once
#include <SFML/System.hpp>

// Simple timer struct
struct Timer {
    int seconds;         // duration in seconds
    sf::Clock clock;     // SFML clock to measure time
    bool done = false;   // has the timer finished?
};
