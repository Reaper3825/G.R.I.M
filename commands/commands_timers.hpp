#pragma once
#include "commands_core.hpp"
#include "timer.hpp"
#include <vector>

// Host-owned timer queue; no callback or state is retained in a plugin.
namespace GRIM::UserTimers {
void schedule(int seconds, const std::string& message);
}

// Check timers for expiration (call this in main loop)
std::vector<CommandResult> checkExpiredTimers();
