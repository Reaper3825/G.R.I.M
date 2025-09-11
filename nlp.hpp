#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <regex>   // <-- needed for std::regex

struct Slot {
    std::string name;
    std::string value;
};

struct Intent {
    std::string name;                       // e.g., "open_app"
    float score = 0.0f;                     // 0..1, for tie-breaking
    std::unordered_map<std::string, std::string> slots; // {"app":"notepad"}
    bool matched = false;
};

class NLP {
public:
    // Load rules from JSON on disk (hot-reload friendly)
    bool load_rules(const std::string& json_path, std::string* error_out = nullptr);

    // Load rules directly from a JSON string (for portable/system hybrid mode)
    bool load_rules_from_string(const std::string& jsonStr, std::string* error_out = nullptr);

    // Parse user text into Intent
    Intent parse(const std::string& text) const;

    // For debugging: list the known intent names
    std::vector<std::string> list_intents() const;

private:
    struct Rule {
        std::string intent;
        std::string description; // optional doc string

        std::string pattern_str; // keep raw regex string (good for debugging/logging)
        std::regex pattern;      // compiled regex used at runtime

        std::vector<std::string> slot_names; // capture-group names
        float score_boost = 0.0f;
        bool case_insensitive = true;
    };

    std::vector<Rule> rules_;

    //private helper to unify parsing logic
    bool parse_rules(const std::string& txt, std::string* error_out);
};
