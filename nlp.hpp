#pragma once
#include <string>
#include <vector>
#include <regex>
#include "intent.hpp"   // <-- add this

class NLP {
public:
    struct Rule {
        std::string intent;
        std::string pattern_str;
        std::regex pattern;
        double boost = 0.0;  // optional confidence boost
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
