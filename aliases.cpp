//aliases.cpp
#include "aliases.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <nlohmann/json.hpp> // you already have this for ai.hpp

std::unordered_map<std::string, std::string> appAliases;

bool loadAliases(const std::string& path) {
    try {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[WARN] Could not open alias file: " << path << "\n";
            return false;
        }
        nlohmann::json j;
        f >> j;

        appAliases.clear();
        for (auto it = j.begin(); it != j.end(); ++it) {
            appAliases[it.key()] = it.value();
        }

        std::cerr << "[INFO] Loaded " << appAliases.size() << " aliases from " << path << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load aliases: " << e.what() << "\n";
        return false;
    }
}
