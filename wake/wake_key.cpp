#include "wake_key.hpp"
#include "wake.hpp"     // so we can access Wake::g_awake
#include "logger.hpp"

#include <SFML/Window/Keyboard.hpp>

namespace WakeKey {
    void update() {
        // Example hotkey: F9 wakes GRIM
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F9)) {
            if (!Wake::g_awake.load()) {
                Wake::g_awake.store(true);
                logTrace("WakeKey", "F9 pressed - waking GRIM");
            }
        }

        // Example hotkey: F10 puts GRIM back to sleep
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F10)) {
            if (Wake::g_awake.load()) {
                Wake::g_awake.store(false);
                logTrace("WakeKey", "F10 pressed - putting GRIM to sleep");
            }
        }
    }
}
