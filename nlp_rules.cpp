#include "nlp_rules.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <regex>
#include <algorithm>

static std::vector<NlpRule> g_rules;

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

bool loadNlpRules(const fs::path& path) {
    g_rules.clear();
    std::ifstream f(path);
    if (!f.is_open()) return false;

    nlohmann::json j; f >> j;
    if (!j.is_array()) return false;

    for (const auto& item : j) {
        NlpRule r;
        if (item.contains("id")) r.id = item["id"].get<std::string>();
        if (item.contains("keywords")) {
            for (auto& kw : item["keywords"]) r.keywords.push_back(toLower(kw.get<std::string>()));
        }
        if (item.contains("regex")) r.regex = item["regex"].get<std::string>();
        if (item.contains("command")) r.command = item["command"].get<std::string>();
        if (item.contains("defaults")) {
            for (auto it = item["defaults"].begin(); it != item["defaults"].end(); ++it) {
                r.defaults.emplace_back(it.key(), it.value().get<std::string>());
            }
        }
        if (!r.command.empty()) g_rules.push_back(std::move(r));
    }
    return true;
}

static std::string applyDefaults(std::string cmd, const std::vector<std::pair<std::string,std::string>>& defs) {
    for (auto& kv : defs) {
        const std::string key = "{" + kv.first + "}";
        size_t pos = 0;
        while ((pos = cmd.find(key, pos)) != std::string::npos) {
            cmd.replace(pos, key.size(), kv.second);
            pos += kv.second.size();
        }
    }
    return cmd;
}

std::string mapTextToCommand(const std::string& inputLowered) {
    // 1) Try regex rules first (more specific)
    for (const auto& r : g_rules) {
        if (r.regex.empty()) continue;
        try {
            std::regex re(r.regex, std::regex::icase);
            std::smatch m;
            if (std::regex_search(inputLowered, m, re)) {
                std::string cmd = r.command;
                // Replace {1}, {2}, ...
                for (size_t i = 1; i < m.size(); ++i) {
                    std::string key = "{" + std::to_string(i) + "}";
                    size_t pos = 0;
                    while ((pos = cmd.find(key, pos)) != std::string::npos) {
                        cmd.replace(pos, key.size(), m[i].str());
                        pos += m[i].str().size();
                    }
                }
                // Apply named defaults if any
                cmd = applyDefaults(cmd, r.defaults);
                return cmd;
            }
        } catch (...) {
            // bad regex; skip
        }
    }

    // 2) Then keyword rules (any keyword contained)
    for (const auto& r : g_rules) {
        if (r.keywords.empty()) continue;
        for (const auto& kw : r.keywords) {
            if (inputLowered.find(kw) != std::string::npos) {
                std::string cmd = r.command;
                cmd = applyDefaults(cmd, r.defaults);
                return cmd;
            }
        }
    }

    return "";
}
