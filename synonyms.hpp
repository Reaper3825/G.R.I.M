#pragma once
#include <string>
#include <unordered_map>

// Load synonyms from synonyms.json into a map
bool loadSynonyms(const std::string& path);

// Normalize a word to its canonical form (returns input if no match)
std::string normalizeWord(const std::string& input);
