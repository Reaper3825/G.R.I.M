#include "NLP.hpp"
#include <nlohmann/json.hpp>
#include <regex>
#include <fstream>
#include <sstream>
#include <iostream>

// --- Internal helper: parse rules from JSON ---
bool NLP::parse_rules(const std::string& txt, std::string* error_out) {
    rules_.clear();

    try {
        auto j = nlohmann::json::parse(txt);

        if (!j.is_array()) {
            if (error_out) *error_out = "Rules JSON must be an array";
            std::cerr << "[NLP] parse_rules: JSON was not an array\n";
            return false;
        }

        for (auto& item : j) {
            if (!item.is_object()) continue;

            Rule r;
            r.intent           = item.value("intent", "");
            r.description      = item.value("description", "");
            r.pattern          = item.value("pattern", "");
            r.slot_names       = item.value("slot_names", std::vector<std::string>{});
            r.score_boost      = item.value("score_boost", 0.0f);
            r.case_insensitive = item.value("case_insensitive", true);

            if (!r.intent.empty() && !r.pattern.empty()) {
                std::cerr << "[NLP] Loaded rule: intent=" << r.intent
                          << " pattern=" << r.pattern
                          << " slots=" << r.slot_names.size()
                          << " boost=" << r.score_boost << "\n";
                rules_.push_back(std::move(r));
            } else {
                std::cerr << "[NLP] Skipped invalid rule (missing intent or pattern)\n";
            }
        }
    } catch (const std::exception& e) {
        if (error_out) *error_out = e.what();
        std::cerr << "[NLP] Failed to parse rules JSON: " << e.what() << "\n";
        return false;
    }

    if (rules_.empty()) {
        if (error_out) *error_out = "No valid rules found.";
        std::cerr << "[NLP] parse_rules: no rules parsed\n";
        return false;
    }

    std::cerr << "[NLP] parse_rules: loaded " << rules_.size() << " rules total\n";
    return true;
}

// --- Public APIs ---

bool NLP::load_rules(const std::string& json_path, std::string* error_out) {
    std::ifstream f(json_path);
    if (!f) {
        if (error_out) *error_out = "Failed to open " + json_path;
        std::cerr << "[NLP] load_rules: could not open file " << json_path << "\n";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_rules(ss.str(), error_out);
}

bool NLP::load_rules_from_string(const std::string& jsonStr, std::string* error_out) {
    return parse_rules(jsonStr, error_out);
}

Intent NLP::parse(const std::string& text) const {
    Intent best;
    for (const auto& r : rules_) {
        std::regex_constants::syntax_option_type flags = std::regex_constants::ECMAScript;
        if (r.case_insensitive) {
            flags = static_cast<std::regex_constants::syntax_option_type>(flags | std::regex_constants::icase);
        }

        try {
            std::regex rx(r.pattern, flags);
            std::smatch m;
            if (std::regex_search(text, m, rx)) {
                Intent cur;
                cur.name = r.intent;
                cur.matched = true;
                cur.score = 0.5f + r.score_boost;

                // Capture groups → slots
                for (size_t gi = 1; gi < m.size() && gi-1 < r.slot_names.size(); ++gi) {
                    cur.slots[r.slot_names[gi-1]] = m[gi].str();
                }

                if (!best.matched || cur.score > best.score) {
                    best = cur;
                }

                // Debug log
                std::cerr << "[NLP] Matched intent=" << cur.name
                          << " score=" << cur.score
                          << " slots=" << cur.slots.size() << "\n";
            }
        } catch (const std::regex_error& ex) {
            std::cerr << "[REGEX ERROR] intent=" << r.intent
                      << " pattern=" << r.pattern
                      << " error=" << ex.what() << "\n";
        }
    }

    if (!best.matched) {
        std::cerr << "[NLP] No intent matched for input: \"" << text << "\"\n";
    }
    return best;
}

std::vector<std::string> NLP::list_intents() const {
    std::vector<std::string> out;
    out.reserve(rules_.size());
    for (auto& r : rules_) out.push_back(r.intent);
    return out;
}
