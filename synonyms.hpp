#pragma once
#include <string>
#include <unordered_map>

// Load synonyms from a JSON file into the map
bool loadSynonyms(const std::string& path);

// Load synonyms directly from a JSON string (portable/system hybrid mode)
bool loadSynonymsFromString(const std::string& jsonStr);

// Normalize a word to its canonical form (returns input if no match)
std::string normalizeWord(const std::string& input);
