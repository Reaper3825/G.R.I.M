#pragma once
#include "commands_core.hpp"

// Task execution commands
CommandResult cmdExecuteTask(const std::string& arg);
CommandResult cmdTaskStatus(const std::string& arg);
CommandResult cmdTaskCancel(const std::string& arg);
