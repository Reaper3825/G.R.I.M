#include "aliases.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <algorithm>

static std::unordered_map<std::string, std::string> aliasMap;

// ------------------------------------------------------------
// Simple Levenshtein distance (edit distance) implementation
// ------------------------------------------------------------
static int levenshtein(const std::string& s1, const std::string& s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();

    std::vector<int> prev(n + 1), curr(n + 1);

    for (size_t j = 0; j <= n; j++) {
        prev[j] = static_cast<int>(j);
    }

    for (size_t i = 1; i <= m; i++) {
        curr[0] = static_cast<int>(i);
        for (size_t j = 1; j <= n; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            curr[j] = std::min({
                prev[j] + 1,        // deletion
                curr[j - 1] + 1,    // insertion
                prev[j - 1] + cost  // substitution
            });
        }
        prev.swap(curr);
    }
    return prev[n];
}

void loadAliases(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "[WARN] Could not open aliases file: " << filename << "\n";
        return;
    }

    try {
        nlohmann::json j;
        in >> j;

        aliasMap.clear();
        for (auto& [k, v] : j.items()) {
            aliasMap[k] = v.get<std::string>();
        }

        std::cerr << "[INFO] Aliases loaded from file. "
                  << aliasMap.size() << " entries.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse aliases file: "
                  << e.what() << "\n";
    }
}

void loadAliasesFromString(const std::string& jsonStr) {
    try {
        nlohmann::json j = nlohmann::json::parse(jsonStr);

        aliasMap.clear();
        for (auto& [k, v] : j.items()) {
            aliasMap[k] = v.get<std::string>();
        }

        std::cerr << "[INFO] Aliases loaded from string. "
                  << aliasMap.size() << " entries.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse aliases string: "
                  << e.what() << "\n";
    }
}

std::string resolveAlias(const std::string& name) {
    // 1. Exact match
    auto it = aliasMap.find(name);
    if (it != aliasMap.end()) {
        return it->second;
    }

    // 2. Fuzzy match
    int bestDistance = 999;
    std::string bestMatch;

    for (const auto& [key, value] : aliasMap) {
        int dist = levenshtein(name, key);
        if (dist < bestDistance) {
            bestDistance = dist;
            bestMatch = value;
        }
    }

    // 3. Threshold check (tune as needed)
    if (bestDistance <= 2 && !bestMatch.empty()) {
        std::cerr << "[Alias] Fuzzy matched '" << name
                  << "' → '" << bestMatch
                  << "' (distance=" << bestDistance << ")\n";
        return bestMatch;
    }

    // No match
    return {};
}
