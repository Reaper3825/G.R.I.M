#pragma once
#include <string>
#include "commands_interface.hpp"

CommandResult cmdSystemInfo(const std::string& arg);
CommandResult cmdTestIntent(const std::string& arg);

// ? NEW: NLP management commands
CommandResult cmdNlpStats(const std::string& arg);
CommandResult cmdNlpLearn(const std::string& arg);
CommandResult cmdNlpSave(const std::string& arg);
CommandResult cmdTestNlp(const std::string& arg);
