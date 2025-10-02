#pragma once
#include <string>
#include <cstddef>

// Load rules from a JSON file.
// - Never throws; returns true if rules loaded and non-empty.
// - Safe to call multiple times; later calls replace existing rules.
bool loadNlpRules(const std::string& path);

// Reload using the last successful path.
// - Returns true if reload succeeded, false otherwise.
bool reloadNlpRules() noexcept;

// Returns true if at least one rule is loaded.
bool nlpRulesLoaded() noexcept;

// Returns the last successfully loaded rules file path (or empty string).
std::string nlpRulesPath();

// Map a LOWERCASED input string to a command string.
// - Returns empty string if no rule matched.
// - This is what your NLP layer calls.
std::string mapTextToCommand(const std::string& lowered);

// (Optional) Debug helper: also returns which rule matched.
// - If 'applied' is non-null and a rule matches, it is filled in.
struct NlpAppliedRule {
    std::string match;
    std::string command;
    bool prefix = false; // true if the rule was a "prefix" rule (e.g., "go to ")
};
bool mapTextToCommandDebug(const std::string& lowered,
                           std::string& outCommand,
                           NlpAppliedRule* applied = nullptr);

// Number of currently loaded rules (for sanity checks).
std::size_t nlpRuleCount() noexcept;
