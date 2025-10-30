#pragma once
#include "commands_core.hpp"

// Perception/Vision query commands
CommandResult cmdPerceptionWhat(const std::string& arg);
CommandResult cmdPerceptionRead(const std::string& arg);
CommandResult cmdPerceptionDetect(const std::string& arg);
