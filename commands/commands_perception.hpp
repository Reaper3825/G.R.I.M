#pragma once
#include "commands_core.hpp"

// Perception/Vision query commands
CommandResult cmdPerceptionWhat(const std::string& arg);
CommandResult cmdPerceptionRead(const std::string& arg);
CommandResult cmdPerceptionDetect(const std::string& arg);

// Input control commands
CommandResult cmdInputMoveMouse(const std::string& arg);
CommandResult cmdInputClick(const std::string& arg);
CommandResult cmdInputType(const std::string& arg);
CommandResult cmdInputKey(const std::string& arg);
CommandResult cmdInputClickOn(const std::string& arg);
