#include "synonyms.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// ---------------- Globals ----------------

// Canonical synonym mapping: synonym word -> canonical command
static std::unordered_map<std::string, std::string> synonymMap;

// Full list of synonyms (canonical -> list of words)
std::unordered_map<std::string, std::vector<std::string>> g_synonyms;

// Words that trigger transcript completion
std::vector<std::string> g_completionTriggers;


// ---------------- Helpers ----------------
static void loadFromJson(const nlohmann::json& j) {
    synonymMap.clear();
    g_synonyms.clear();
    g_completionTriggers.clear();

    for (auto& [key, value] : j.items()) {
        if (key == "completion_triggers") {
            // Load completion triggers
            g_completionTriggers = value.get<std::vector<std::string>>();
            for (auto& w : g_completionTriggers) {
                std::transform(w.begin(), w.end(), w.begin(), ::tolower);
            }
        } else {
            // Load synonyms for canonical word
            std::vector<std::string> words = value.get<std::vector<std::string>>();
            for (auto& w : words) {
                std::string lw = w;
                std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
                synonymMap[lw] = key; // map synonym -> canonical
            }
            g_synonyms[key] = words;
        }
    }

    std::cerr << "[INFO] Synonyms loaded: " << synonymMap.size()
              << " entries, " << g_completionTriggers.size()
              << " completion triggers.\n";
}


// ---------------- API ----------------
bool loadSynonyms(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[WARN] Could not load " << path << "\n";
        return false;
    }

    try {
        nlohmann::json j;
        f >> j;
        loadFromJson(j);
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Failed to parse synonyms: " << ex.what() << "\n";
        return false;
    }
}

bool loadSynonymsFromString(const std::string& jsonStr) {
    try {
        nlohmann::json j = nlohmann::json::parse(jsonStr);
        loadFromJson(j);
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Failed to parse synonyms string: "
                  << ex.what() << "\n";
        return false;
    }
}

// Normalize a word to its canonical form (returns input if no match)
std::string normalizeWord(const std::string& input) {
    std::string word = input;
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);

    auto it = synonymMap.find(word);
    if (it != synonymMap.end()) {
        return it->second; // return canonical form
    }
    return input; // return original if no match
}
