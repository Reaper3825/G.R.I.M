#pragma once
#include <string>
#include <unordered_map>
#include <vector>

// ---------------- Loaders ----------------

// Load synonyms from a JSON file into memory
bool loadSynonyms(const std::string& path);

// Load synonyms directly from a JSON string (portable/system hybrid mode)
bool loadSynonymsFromString(const std::string& jsonStr);

// Normalize a word to its canonical form (returns input if no match)
std::string normalizeWord(const std::string& input);


// ---------------- Globals ----------------

// Full synonym map: canonical -> list of synonyms
extern std::unordered_map<std::string, std::vector<std::string>> g_synonyms;

// Transcript completion trigger words
extern std::vector<std::string> g_completionTriggers;
