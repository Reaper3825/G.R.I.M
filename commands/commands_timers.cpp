#include "commands_timers.hpp"
#include "error_manager.hpp"
#include "resources.hpp"   // globals: timers, history
#include <sstream>
#include <chrono>

// ====================================================
// [Timer] Set a new timer (arg = number of seconds)
// ====================================================
namespace GRIM::UserTimers {
void schedule(int seconds, const std::string& message) {
    Timer timer;
    timer.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    timer.message = message;
    timers.push_back(std::move(timer));
}
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
