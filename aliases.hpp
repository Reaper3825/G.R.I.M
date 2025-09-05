#pragma once
#include <string>

// Resolve an alias → returns mapped value, or empty if none
std::string resolveAlias(const std::string& name);

// Load aliases from a JSON file
void loadAliases(const std::string& filename);

// Load aliases directly from a JSON string (for portable/system hybrid mode)
void loadAliasesFromString(const std::string& jsonStr);
