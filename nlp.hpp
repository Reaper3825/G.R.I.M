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
        std::string intent;
        std::string pattern_str;
        std::regex pattern;
        double boost = 0.0;

        // 🔹 Needed for slot extraction
        std::vector<std::string> slot_names;
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
