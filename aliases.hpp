//aliases.hpp
#pragma once
#include <string>
#include <unordered_map>

// Global alias dictionary
extern std::unordered_map<std::string, std::string> appAliases;

// Load aliases from JSON file
bool loadAliases(const std::string& path);
