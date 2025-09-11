#include "nlp.hpp"
#include <regex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 🔹 Global NLP object definition (only needed if you want global usage)
NLP g_nlp;

// ---------------------- Intent Parsing ----------------------

Intent NLP::parse(const std::string& text) const {
    Intent best;
    double bestScore = 0.0;

    for (const auto& rule : rules) {
        std::smatch match;
        if (std::regex_match(text, match, rule.pattern)) {
            Intent intent;
            intent.name = rule.intent;
            intent.matched = true;

            // base confidence
            intent.score = 0.5;

            // optional boost from rule
            if (rule.boost > 0.0) {
                intent.score += rule.boost;
            }

            // slot extraction if regex groups exist
            for (size_t i = 1; i < match.size(); i++) {
                intent.slots["slot" + std::to_string(i)] = match[i].str();
            }

            // track best match
            if (intent.score > bestScore) {
                best = intent;
                bestScore = intent.score;
            }
        }
    }

    return best;
}

// ---------------------- Rule Loading ----------------------

bool NLP::load_rules_from_string(const std::string& rulesText, std::string* err) {
    try {
        auto data = json::parse(rulesText);
        rules.clear();

        for (auto& r : data) {
            Rule rule;
            rule.intent           = r.value("intent", "");
            rule.pattern_str      = r.value("pattern", "");
            rule.boost            = r.value("boost", 0.0);
            bool caseInsensitive  = r.value("case_insensitive", false);

            try {
                rule.pattern = std::regex(
                    rule.pattern_str,
                    caseInsensitive ? std::regex::icase : std::regex::ECMAScript
                );
            } catch (const std::exception& e) {
                if (err) *err = "Invalid regex for intent '" + rule.intent + "': " + e.what();
                return false;
            }

            rules.push_back(rule);  // ✅ fixed
        }

        std::cout << "[NLP] Loaded " << rules.size() << " rules\n";  // ✅ fixed
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool NLP::load_rules(const std::string& json_path, std::string* error_out) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        if (error_out) *error_out = "Could not open rules file: " + json_path;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return this->load_rules_from_string(buffer.str(), error_out);
}
