#include "wake_key.hpp"
#include "logger.hpp"

#include <SFML/Window/Keyboard.hpp>

namespace WakeKey {
    void update() {
        // Example hotkey: F9 wakes GRIM
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F9)) {
            logTrace("WakeKey", "F9 pressed - waking GRIM");
            
        }
    }
}
