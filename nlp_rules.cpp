#include "nlp.hpp"

// ------------------------------------------------------------
// Load NLP rules from a JSON file into the global g_nlp object
// ------------------------------------------------------------
bool loadNlpRules(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Could not open NLP rules file: " << path << "\n";
        return false;
    }

    try {
        nlohmann::json j;
        in >> j;

        if (!j.is_array()) {
            std::cerr << "[ERROR] Invalid NLP rules JSON (expected array)\n";
            return false;
        }

        // Convert the parsed JSON back to string
        std::string rulesText = j.dump();

        std::string err;
        if (!g_nlp.load_rules_from_string(rulesText, &err)) {
            std::cerr << "[ERROR] Failed to load NLP rules: " << err << "\n";
            return false;
        }

        std::cerr << "[NLP] Loaded " << g_nlp.rule_count()
                  << " rules from " << path << "\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse NLP rules: " << e.what() << "\n";
        return false;
    }
}
