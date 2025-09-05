#include "aliases.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <iostream>

static std::unordered_map<std::string, std::string> aliasMap;

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
    auto it = aliasMap.find(name);
    if (it != aliasMap.end()) {
        return it->second;
    }
    return {};
}
