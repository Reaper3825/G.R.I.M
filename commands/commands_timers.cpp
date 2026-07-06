#include "commands_timers.hpp"
#include "error_manager.hpp"
#include "resources.hpp"   // globals: timers, history
#include <sstream>
#include <chrono>

// ====================================================
// [Timer] Set a new timer (arg = number of seconds)
// ====================================================
CommandResult cmdSetTimer(const std::string& arg) {
    int seconds = 0;
    try {
        seconds = std::stoi(arg);
    } catch (...) {
        return {
            false,                                          // success
            "[Timer][Error] Invalid number of seconds.",   // message
            "ERR_TIMER_INVALID",                            // errorCode
            "error",                                        // category
            "Invalid timer duration",                       // voice
            Colors::Red                                     // color
        };
    }

    if (seconds <= 0) {
        return {
            false,                                              // success
            "[Timer][Error] Duration must be positive.",        // message
            "ERR_TIMER_NONPOSITIVE",                            // errorCode
            "error",                                            // category
            "Non-positive timer duration",                      // voice
            Colors::Red                                         // color
        };
    }

    // Create timer with expiry time and message
    Timer t;
    t.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    t.message = "Timer expired after " + std::to_string(seconds) + "s";

    timers.push_back(t);

    return {
        true,                                                   // success
        "[Timer] Set for " + std::to_string(seconds) + " seconds.", // message
        "ERR_NONE",                                             // errorCode
        "routine",                                              // category
        "Timer set",                                            // voice
        Colors::Green                                           // color
    };
}

// ====================================================
// [Timer] Check timers for expiration
// ====================================================
std::vector<CommandResult> checkExpiredTimers() {
    std::vector<CommandResult> results;
    auto now = std::chrono::steady_clock::now();

    auto it = timers.begin();
    while (it != timers.end()) {
        if (now >= it->expiry) {
            results.push_back({
                true,                           // success
                "[Timer] " + it->message,       // message
                "ERR_NONE",                     // errorCode
                "routine",                      // category
                "Timer expired",                // voice
                Colors::Yellow                  // color
            });
            it = timers.erase(it);
        } else {
            ++it;
        }
    }

    return results;
}
