#pragma once
#include <string>

// Forward declaration
struct CommandResult;

// OSINT Self-Audit Commands
CommandResult cmdProfileSelf(const std::string& args);
CommandResult cmdSherlockSweep(const std::string& args);
CommandResult cmdOsintReport(const std::string& args);
CommandResult cmdOsintStatus(const std::string& args);
CommandResult cmdOsintClearCache(const std::string& args);

