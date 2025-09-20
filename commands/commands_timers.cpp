#include "commands_timers.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "resources.hpp"   // globals: timers, history
#include <sstream>
#include <chrono>

// ------------------------------------------------------------
// [Timer] Set a new timer (arg = number of seconds)
// ------------------------------------------------------------
CommandResult cmdSetTimer(const std::string& arg) {
    int seconds = 0;
    try {
        seconds = std::stoi(arg);
    } catch (...) {
        return {
            "[Timer][Error] Invalid number of seconds.",
            false,
            sf::Color::Red,
            "ERR_TIMER_INVALID",
            "Invalid timer duration",
            "error"
        };
    }

    if (seconds <= 0) {
        return {
            "[Timer][Error] Duration must be positive.",
            false,
            sf::Color::Red,
            "ERR_TIMER_NONPOSITIVE",
            "Non-positive timer duration",
            "error"
        };
    }

    // Create timer with expiry time and message
    Timer t;
    t.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    t.message = "Timer expired after " + std::to_string(seconds) + "s";

    timers.push_back(t);

    return {
        "[Timer] Set for " + std::to_string(seconds) + " seconds.",
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Timer set",
        "routine"
    };
}

// ------------------------------------------------------------
// [Timer] Check timers for expiration
// ------------------------------------------------------------
std::vector<CommandResult> checkExpiredTimers() {
    std::vector<CommandResult> results;
    auto now = std::chrono::steady_clock::now();

    auto it = timers.begin();
    while (it != timers.end()) {
        if (now >= it->expiry) {
            results.push_back({
                "[Timer] " + it->message,
                true,
                sf::Color::Yellow,
                "ERR_NONE",
                "Timer expired",
                "routine"
            });
            it = timers.erase(it);
        } else {
            ++it;
        }
    }

    return results;
}
