#include "nlp.hpp"
#include "resources.hpp"   // 👈 for getResourcePath()
#include <regex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <algorithm>
#include <cctype>

using json = nlohmann::json;
namespace fs = std::filesystem;

// 🔹 Global NLP object definition
NLP g_nlp;

// ------------------------------------------------------------
// Utility: normalize input (lowercase + strip punctuation + trim)
// ------------------------------------------------------------
static std::string normalizeInput(const std::string& input) {
    std::string out;
    out.reserve(input.size());

    for (char c : input) {
        if (!std::ispunct(static_cast<unsigned char>(c))) {
            out.push_back(std::tolower(static_cast<unsigned char>(c)));
        }
    }

    // 🔹 Trim leading/trailing spaces
    auto start = out.find_first_not_of(' ');
    auto end   = out.find_last_not_of(' ');
    if (start == std::string::npos) {
        return ""; // all spaces
    }
    return out.substr(start, end - start + 1);
}

// ---------------------- Intent Parsing ----------------------

Intent NLP::parse(const std::string& rawText) const {
    // ✅ Normalize input before applying regex
    std::string text = normalizeInput(rawText);
    std::cout << "[DEBUG][NLP] Raw: \"" << rawText
              << "\" → Normalized: \"" << text << "\"\n";

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

            // 🔹 Debug: show which rule matched
            std::cout << "[DEBUG][NLP] Matched intent: " << intent.name
                      << " (pattern: \"" << rule.pattern_str << "\")"
                      << " score=" << intent.score << "\n";

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
            rule.boost            = r.value("score_boost", 0.0);
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

            rules.push_back(rule);
        }

        std::cout << "[NLP] Loaded " << rules.size() << " rules\n";
        return true;
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }
}

bool NLP::load_rules(const std::string& filename, std::string* error_out) {
    // Build primary + fallback paths
    fs::path primary   = fs::path(getResourcePath()) / filename;                    // build/resources
    fs::path secondary = fs::current_path().parent_path() / "resources" / filename; // ../resources

    std::ifstream file(primary);
    if (!file.is_open()) {
        // try fallback
        file.open(secondary);
        if (!file.is_open()) {
            if (error_out) {
                *error_out = "Could not open rules file. Tried: " +
                             primary.string() + " and " + secondary.string();
            }
            std::cerr << "[NLP] Failed to load rules. Tried: "
                      << primary << " and " << secondary << std::endl;
            return false;
        } else {
            std::cout << "[NLP] Loaded rules from fallback path: " 
                      << secondary << std::endl;
        }
    } else {
        std::cout << "[NLP] Loaded rules from: " << primary << std::endl;
    }

    // Read entire file into buffer
    std::stringstream buffer;
    buffer << file.rdbuf();
    return this->load_rules_from_string(buffer.str(), error_out);
}

// ---------------------- Reload Wrapper ----------------------

#include "console_history.hpp"
#include "error_manager.hpp"
#include <SFML/Graphics.hpp>

// Externals
extern ConsoleHistory history;

CommandResult reloadNlpRules() {
    std::string error;
    if (g_nlp.load_rules("nlp_rules.json", &error)) {
        std::cout << "[NLP] Rules reloaded successfully.\n";
        return {
            "[NLP] Rules reloaded successfully.",
            true,
            sf::Color::Green,
            "" // no error code
        };
    } else {
        std::cerr << "[NLP] Reload failed: " << error << "\n";
        return {
            ErrorManager::getUserMessage("ERR_NLP_RELOAD_FAIL") + ": " + error,
            false,
            sf::Color::Red,
            "ERR_NLP_RELOAD_FAIL"
        };
    }
}
