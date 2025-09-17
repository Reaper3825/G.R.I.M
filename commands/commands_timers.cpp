#include "commands_timers.hpp"
#include "response_manager.hpp"
#include "error_manager.hpp"

#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <cctype>
#include <algorithm>

// Externals
extern std::vector<Timer> timers;

// ------------------------------------------------------------
// Helper: parse duration like "90", "5m", "2h30m", "45s"
// ------------------------------------------------------------
static int parseDuration(const std::string& arg) {
    int totalSeconds = 0;
    std::string num;
    for (size_t i = 0; i < arg.size(); ++i) {
        if (std::isdigit(arg[i])) {
            num += arg[i];
        } else {
            if (num.empty()) continue;

            int value = std::stoi(num);
            char unit = std::tolower(arg[i]);

            if (unit == 'h') totalSeconds += value * 3600;
            else if (unit == 'm') totalSeconds += value * 60;
            else if (unit == 's') totalSeconds += value;
            else {
                // Unknown unit → treat as seconds
                totalSeconds += value;
            }
            num.clear();
        }
    }

    // If no units provided, treat whole thing as seconds
    if (!num.empty()) {
        totalSeconds += std::stoi(num);
    }

    return totalSeconds;
}

// ------------------------------------------------------------
// [Timer] Set a timer
// ------------------------------------------------------------
CommandResult cmdSetTimer(const std::string& arg) {
    if (arg.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_TIMER_MISSING_VALUE"),
            false,
            sf::Color::Red,
            "ERR_TIMER_MISSING_VALUE",
            "Timer value required",
            "error"
        };
    }

    try {
        int seconds = parseDuration(arg);
        if (seconds <= 0) {
            return {
                ErrorManager::getUserMessage("ERR_TIMER_INVALID_VALUE") + ": " + arg,
                false,
                sf::Color::Red,
                "ERR_TIMER_INVALID_VALUE",
                "Invalid timer value",
                "error"
            };
        }

        Timer t;
        t.seconds = seconds;
        timers.push_back(t);

        return {
            "[Timer] Timer set for " + std::to_string(seconds) + " seconds.",
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Timer set for " + std::to_string(seconds) + " seconds",
            "routine"
        };
    } catch (const std::exception&) {
        return {
            ErrorManager::getUserMessage("ERR_TIMER_INVALID_VALUE") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_TIMER_INVALID_VALUE",
            "Invalid timer value",
            "error"
        };
    }
}

// ------------------------------------------------------------
// [Timer] Check for expired timers
//   (Call this periodically in main loop)
// ------------------------------------------------------------
std::vector<CommandResult> checkExpiredTimers() {
    std::vector<CommandResult> results;

    for (auto& t : timers) {
        if (!t.done && t.clock.getElapsedTime().asSeconds() >= t.seconds) {
            t.done = true;
            results.push_back({
                "[Timer] Time's up! (" + std::to_string(t.seconds) + "s)",
                true,
                sf::Color::Yellow,
                "ERR_NONE",
                "Time's up",
                "routine"
            });
        }
    }

    return results;
}
