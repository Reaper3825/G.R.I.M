#pragma once
#include "commands_core.hpp"

// Input control commands
CommandResult cmdInputMoveMouse(const std::string& arg);
CommandResult cmdInputClick(const std::string& arg);
CommandResult cmdInputType(const std::string& arg);
CommandResult cmdInputKey(const std::string& arg);
CommandResult cmdInputClickOn(const std::string& arg);
