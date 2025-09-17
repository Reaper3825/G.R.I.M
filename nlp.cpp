#include "nlp.hpp"
#include "intent.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "console_history.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

NLP g_nlp;

// ------------------------------------------------------------
// Parse text against loaded NLP rules
// ------------------------------------------------------------
Intent NLP::parse(const std::string& text) const {
    Intent intent;
    intent.matched = false;

    for (const auto& rule : rules) {
        std::smatch match;
        if (std::regex_match(text, match, rule.pattern)) {
            intent.matched = true;
            intent.name = rule.intent;
            intent.description = rule.description;
            intent.category = rule.category.empty() ? "general" : rule.category;
            intent.confidence = 0.5 + rule.score_boost; // base + boost

            // Map regex captures to slots
            for (size_t i = 1; i < match.size() && i <= rule.slot_names.size(); i++) {
                intent.slots[rule.slot_names[i - 1]] = match[i].str();
            }
            return intent;
        }
    }

    // No rule matched
    return intent;
}

// ------------------------------------------------------------
// Load rules from file
// ------------------------------------------------------------
bool NLP::load_rules(const std::string& path, std::string* err) {
    try {
        std::ifstream f(path);
        if (!f) {
            if (err) *err = "Could not open file: " + path;
            return false;
        }
        nlohmann::json j;
        f >> j;
        f.close();

        rules.clear();

        for (auto& r : j) {
            Rule rule;
            rule.intent = r.value("intent", "");
            rule.description = r.value("description", "");
            rule.pattern_str = r.value("pattern", "");
            rule.slot_names = r.value("slot_names", std::vector<std::string>{});
            rule.score_boost = r.value("score_boost", 0.0);
            rule.case_insensitive = r.value("case_insensitive", true);
            rule.category = r.value("category", "general");

            try {
                std::regex::flag_type flags = std::regex::ECMAScript;
                if (rule.case_insensitive) {
                    flags |= std::regex::icase;
                }
                rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (std::exception& e) {
                std::cerr << "[NLP] Invalid regex for intent " << rule.intent
                          << ": " << e.what() << "\n";
                continue;
            }

            rules.push_back(rule);
        }

        std::cout << "[NLP] Loaded " << rules.size() << " rules from " << path << "\n";
        return true;
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// ------------------------------------------------------------
// Load rules from a JSON string
// ------------------------------------------------------------
bool NLP::load_rules_from_string(const std::string& rulesText, std::string* err) {
    try {
        nlohmann::json j = nlohmann::json::parse(rulesText);

        rules.clear();

        for (auto& r : j) {
            Rule rule;
            rule.intent = r.value("intent", "");
            rule.description = r.value("description", "");
            rule.pattern_str = r.value("pattern", "");
            rule.slot_names = r.value("slot_names", std::vector<std::string>{});
            rule.score_boost = r.value("score_boost", 0.0);
            rule.case_insensitive = r.value("case_insensitive", true);
            rule.category = r.value("category", "general");

            try {
                std::regex::flag_type flags = std::regex::ECMAScript;
                if (rule.case_insensitive) {
                    flags |= std::regex::icase;
                }
                rule.pattern = std::regex(rule.pattern_str, flags);
            } catch (std::exception& e) {
                std::cerr << "[NLP] Invalid regex for intent " << rule.intent
                          << ": " << e.what() << "\n";
                continue;
            }

            rules.push_back(rule);
        }

        std::cout << "[NLP] Loaded " << rules.size() << " rules from string\n";
        return true;
    } catch (std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

// ------------------------------------------------------------
// Reload wrapper
// ------------------------------------------------------------
CommandResult reloadNlpRules() {
    std::string err;
    if (!g_nlp.load_rules(getResourcePath() + "/nlp_rules.json", &err)) {
        return {
            "[Error] Failed to reload NLP rules",
            false,
            sf::Color::Red,
            "ERR_NLP_RELOAD_FAILED"
        };
    }
    return {
        "[NLP] Rules reloaded successfully",
        true,
        sf::Color::Green,
        ""
    };
}
