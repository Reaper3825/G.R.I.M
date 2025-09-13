#pragma once
#include <string>

// ------------------------------------------------------------
// Aliases API
// ------------------------------------------------------------

// Resolve an alias → returns mapped value (e.g., executable path),
// or empty string if none found. Includes fuzzy fallback.
std::string resolveAlias(const std::string& name);

// Load aliases from a JSON file
void loadAliases(const std::string& filename);

// Load aliases directly from a JSON string (for portable/system hybrid mode)
void loadAliasesFromString(const std::string& jsonStr);
