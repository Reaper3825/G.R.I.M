#pragma once
#include <string>
#include <vector>
#include <regex>
#include "intent.hpp"   // defines the Intent struct

// Forward declare to avoid heavy include
struct CommandResult;

class NLP {
public:
    struct Rule {
        std::string intent;        // e.g. "open_app"
        std::string description;   // human-readable ("Open a local application")
        std::string pattern_str;   // raw regex string
        std::regex pattern;        // compiled regex
        double score_boost = 0.0;  // weight to improve ranking
        bool case_insensitive = true; // regex flag

        std::vector<std::string> slot_names; // slot names for regex groups
        std::string category;                // optional grouping (system, app, alias)
    };

    // --- Methods ---
    Intent parse(const std::string& text) const;
    bool load_rules(const std::string& path, std::string* err = nullptr);
    bool load_rules_from_string(const std::string& rulesText, std::string* err = nullptr);

    // --- Debug helper ---
    size_t rule_count() const { return rules.size(); }

private:
    std::vector<Rule> rules;
};

// 🔹 Global NLP object declaration (defined in nlp.cpp)
extern NLP g_nlp;

// ---------------------- Reload Wrapper ----------------------
CommandResult reloadNlpRules();
