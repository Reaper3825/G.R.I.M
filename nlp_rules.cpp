#include "nlp_rules.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <mutex>

// NO namespace here unless header also has it.

struct Rule { std::string match, command; bool prefix = false; };

static std::vector<Rule> g_rules;
static std::mutex g_m;

bool loadNlpRules(const std::string& path) {
    // Minimal stub so you can link; replace with your JSON loader later.
    // Example hardcoded rules so you can test immediately:
    std::lock_guard<std::mutex> lock(g_m);
    g_rules.clear();
    g_rules.push_back({"where am i", "pwd", false});
    g_rules.push_back({"list files", "ls", false});
    g_rules.push_back({"go to ", "cd ", true});
    g_rules.push_back({"open ",  "cd ", true});
    return true;
}

std::string mapTextToCommand(const std::string& lowered) {
    std::lock_guard<std::mutex> lock(g_m);
    for (auto& r : g_rules) {
        if (!r.prefix) {
            if (lowered == r.match) return r.command;
        } else {
            if (lowered.rfind(r.match, 0) == 0) {
                return r.command + lowered.substr(r.match.size());
            }
        }
    }
    return {};
}
