#pragma once
#include <string>
#include "commands_core.hpp" // for CommandResult

// ------------------------------------------------------------
// Alias command handlers
// ------------------------------------------------------------
CommandResult cmdAliasList(const std::string& arg);
CommandResult cmdAliasInfo(const std::string& arg);
CommandResult cmdAliasRefresh(const std::string& arg);
