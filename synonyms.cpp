#include "synonyms.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Internal map: synonym word -> canonical command
static std::unordered_map<std::string, std::string> synonymMap;

bool loadSynonyms(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[WARN] Could not load " << path << "\n";
        return false;
    }

    try {
        nlohmann::json j;
        f >> j;

        synonymMap.clear();
        for (auto& [canonical, words] : j.items()) {
            for (auto& w : words) {
                std::string word = w.get<std::string>();
                // lowercase for safety
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                synonymMap[word] = canonical;
            }
        }

        std::cerr << "[INFO] Synonyms loaded. "
                  << synonymMap.size() << " entries.\n";
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Failed to parse synonyms: " << ex.what() << "\n";
        return false;
    }
}

// ⚠️ This must be **outside** of loadSynonyms
std::string normalizeWord(const std::string& input) {
    std::string word = input;
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);

    auto it = synonymMap.find(word);
    if (it != synonymMap.end()) {
        return it->second; // return canonical form
    }
    return input; // return original if no match
}
