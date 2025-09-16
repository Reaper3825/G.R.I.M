#pragma once
#include "commands_core.hpp"

// ------------------------------------------------------------
// Alias Commands
// ------------------------------------------------------------
// Provides user-facing commands for inspecting and refreshing
// GRIM's alias system (user + auto + fallback).
// ------------------------------------------------------------

CommandResult cmdAliasList(const std::string& args);
CommandResult cmdAliasInfo(const std::string& args);
CommandResult cmdAliasRefresh(const std::string& args);

// Register all alias-related commands
void registerAliasCommands();
