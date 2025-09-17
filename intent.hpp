#pragma once
#include <string>
#include <map>
#include <vector>

// 🔹 Represents the result of NLP parsing
struct Intent {
    std::string name;                     // Intent name, e.g. "open_app"
    std::string description;              // Human-readable description
    std::string category = "general";     // Grouping (system, app, alias, etc.)
    bool matched = false;                 // Did a rule match?

    std::map<std::string, std::string> slots;   // Named slot captures
    std::vector<std::string> groups;      // Raw regex capture groups

    double score = 0.0;                   // Legacy confidence score
    double confidence = 0.0;              // Newer confidence field
};
