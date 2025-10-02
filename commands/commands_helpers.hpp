#pragma once
#include <string>
#include <utility>
#include "nlp/nlp.hpp"   // for Intent struct

// Trim whitespace from both ends
std::string trim(const std::string& s);

// Get a slot value with fallback
std::string getSlot(const Intent& intent, const std::string& name, const std::string& fallback = "");

// Get key/value slot pair
std::pair<std::string, std::string> getKeyValueSlots(const Intent& intent);
