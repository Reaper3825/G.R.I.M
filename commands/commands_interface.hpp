#pragma once
#include "commands_core.hpp"

// Utility/system commands
CommandResult cmdSystemInfo(const std::string& arg);
CommandResult cmdClean(const std::string& arg);
CommandResult cmdShowHelp(const std::string& arg);
CommandResult cmd_reloadNLP(const std::string& arg);
