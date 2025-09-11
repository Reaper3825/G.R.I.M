#include "nlp.hpp"
#include <regex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Parse user input against loaded rules
Intent NLP::parse(const std::string& text) const {
    Intent best;
    double bestScore = 0.0;

    for (const auto& rule : rules_) {
        std::smatch match;
        if (std::regex_match(text, match, rule.pattern)) {
            Intent intent;
            intent.name = rule.intent;
            intent.matched = true;
            intent.score = 0.5; // basic match score

            // slot extraction if regex groups exist
            for (size_t i = 1; i < match.size(); i++) {
                intent.slots["slot" + std::to_string(i)] = match[i].str();
            }

            if (intent.score > bestScore) {
                best = intent;
                bestScore = intent.score;
            }
        }
    }

    return best;
}

// Load rules from JSON string
bool NLP::load_rules_from_string(const std::string& rulesText, std::string* err) {
    try {
        auto data = json::parse(rulesText);
        rules_.clear();

        for (auto& r : data) {
            Rule rule;
            rule.intent = r.value("intent", "");
            rule.pattern_str = r.value("pattern", "");
            rule.pattern = std::regex(rule.pattern_str,
                                      rule.case_insensitive ? std::regex::icase : std::regex::ECMAScript);
            rules_.push_back(rule);
        }
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// Load rules from a JSON file
bool NLP::load_rules(const std::string& json_path, std::string* error_out) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        if (error_out) *error_out = "Could not open rules file: " + json_path;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return load_rules_from_string(buffer.str(), error_out);
}
