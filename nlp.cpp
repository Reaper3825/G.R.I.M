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
#include "commands/commands_core.hpp"

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
        unsigned char uc = static_cast<unsigned char>(c);

        // Keep letters, numbers, spaces, and underscores
        if (std::isalnum(uc) || c == '_' || std::isspace(uc)) {
            out.push_back(std::tolower(uc));
        }
        // drop everything else
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
    best.matched = false;
    double bestScore = -1.0;

    for (const auto& rule : rules) {
        std::smatch match;
        if (std::regex_match(text, match, rule.pattern)) {
            Intent intent;
            intent.name = rule.intent;
            intent.matched = true;

            // Base confidence
            intent.score = 0.5 + rule.boost;

            // Capture raw groups
            intent.groups.clear();
            for (size_t i = 1; i < match.size(); i++) {
                intent.groups.push_back(match[i].str());
            }

            // Map captures to slots
            if (!rule.slot_names.empty()) {
                for (size_t i = 0; i < rule.slot_names.size() && (i + 1) < match.size(); i++) {
                    intent.slots[rule.slot_names[i]] = match[i + 1].str();
                }
            } else {
                // fallback: generic slot1, slot2...
                for (size_t i = 1; i < match.size(); i++) {
                    intent.slots["slot" + std::to_string(i)] = match[i].str();
                }
                // heuristic: common two-slot patterns → verb/app
                if (match.size() == 3) {
                    intent.slots["verb"] = match[1].str();
                    intent.slots["app"]  = match[2].str();
                }
            }

            // 🔹 Debug
            std::cout << "[DEBUG][NLP] Matched intent: " << intent.name
                      << " (pattern: \"" << rule.pattern_str << "\")"
                      << " score=" << intent.score << "\n";
            if (!intent.slots.empty()) {
                for (auto& kv : intent.slots) {
                    std::cout << "   slot[" << kv.first << "]=\"" << kv.second << "\"\n";
                }
            }

            // Select best-scoring intent
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

            // slot_names support
            if (r.contains("slot_names") && r["slot_names"].is_array()) {
                for (auto& sn : r["slot_names"]) {
                    rule.slot_names.push_back(sn.get<std::string>());
                }
            }

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
            "ERR_NONE"
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
