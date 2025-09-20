#pragma once
#include <string>
#include <chrono>

// Expanded timer struct (more flexible than simple SFML clock)
struct Timer {
    std::chrono::steady_clock::time_point expiry;  // when the timer expires
    std::string message;                           // message to display when expired
};
