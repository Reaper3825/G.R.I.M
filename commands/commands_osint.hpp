#pragma once
#include <string>

// Forward declaration
struct CommandResult;

// OSINT Self-Audit Commands
CommandResult cmdProfilePerson(const std::string& args);
CommandResult cmdSherlockSweep(const std::string& args);
CommandResult cmdOsintReport(const std::string& args);
CommandResult cmdOsintStatus(const std::string& args);
CommandResult cmdOsintClearCache(const std::string& args);
// Scan discovered URLs for sensitive data exposure
CommandResult cmdOsintScanSecrets(const std::string& args);  // Scan for sensitive data
CommandResult cmdOsintShowSecrets(const std::string& args);  // View detailed findings
CommandResult cmdOsintShowUI(const std::string& args);       // NEW: Show findings in UI panel

