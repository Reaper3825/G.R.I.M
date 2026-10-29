#pragma once
// Retained host/legacy handlers, not registered agent tools.
// Behavior and possible future uses: docs/RETIRED_AGENT_TOOLS.md.
#include <string>
#include "commands_core.hpp"

// Test commands
CommandResult cmdTestIntent(const std::string& arg);
CommandResult cmdTestNlp(const std::string& arg);

// NLP management commands
CommandResult cmdNlpStats(const std::string& arg);
CommandResult cmdNlpLearn(const std::string& arg);
CommandResult cmdNlpSave(const std::string& arg);

// ? Settings command
CommandResult cmdSettings(const std::string& arg);
