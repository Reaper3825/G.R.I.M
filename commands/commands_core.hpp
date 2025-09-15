#pragma once
#include "commands_core.hpp"
#include "console_history.hpp"

// Externals
extern ConsoleHistory history;

CommandResult cmdClean(const std::string& arg);
CommandResult cmdShowHelp(const std::string& arg);
