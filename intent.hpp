#pragma once
#include <string>
#include <map>
#include <vector>

// 🔹 Represents the result of NLP parsing
struct Intent {
    std::string name;                     // Intent name, e.g. "open_app"
    bool matched = false;                 // Did a rule match?
    std::map<std::string, std::string> slots;   // Named slot captures
    std::vector<std::string> groups;      // Raw regex capture groups
    double score = 0.0;                   // Confidence score
};
