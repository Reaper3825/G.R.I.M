#pragma once
#include "commands_core.hpp"
#include "timer.hpp"

#include <vector>

// Timer commands
CommandResult cmdSetTimer(const std::string& arg);

// Check timers for expiration (call this in main loop)
std::vector<CommandResult> checkExpiredTimers();
