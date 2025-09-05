#include "aliases.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>

static std::unordered_map<std::string, std::string> aliasMap;

void loadAliases(const std::string& filename) {
    std::ifstream in(filename);
    if(!in.is_open()) return;

    nlohmann::json j;
    in >> j;
    for(auto& [k, v] : j.items()) {
        aliasMap[k] = v;
    }
}

std::string resolveAlias(const std::string& name) {
    auto it = aliasMap.find(name);
    if(it != aliasMap.end()) {
        return it->second;
    }
    return {};
}
