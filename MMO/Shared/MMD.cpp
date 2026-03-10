// Multi-Model Orchestration (MMO) - MMD implementation
// Subject tag extraction for subject-based routing.
//======================================================//
#include "MMD.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace GRIM::MMO {

// =========================================================
// Subject keyword → tag mappings
//
// Each keyword maps to a subject tag. When a keyword appears
// in the user's input, the corresponding tag is emitted.
// Tags match ModelInfo::subject_tags for routing.
// =========================================================
static const std::unordered_map<std::string, std::string>& subjectKeywords() {
    static const std::unordered_map<std::string, std::string> kw = {
        // Science & math
        {"equation",      "math"},
        {"calculate",     "math"},
        {"formula",       "math"},
        {"integral",      "math"},
        {"derivative",    "math"},
        {"algebra",       "math"},
        {"geometry",      "math"},
        {"physics",       "science"},
        {"chemistry",     "science"},
        {"biology",       "science"},
        {"molecule",      "science"},
        {"atom",          "science"},
        {"experiment",    "science"},
        {"hypothesis",    "science"},

        // Programming & tech
        {"code",          "programming"},
        {"compile",       "programming"},
        {"debug",         "programming"},
        {"function",      "programming"},
        {"class",         "programming"},
        {"variable",      "programming"},
        {"algorithm",     "programming"},
        {"python",        "programming"},
        {"javascript",    "programming"},
        {"rust",          "programming"},
        {"linux",         "technology"},
        {"windows",       "technology"},
        {"server",        "technology"},
        {"network",       "technology"},
        {"database",      "technology"},

        // Language & writing
        {"translate",     "language"},
        {"grammar",       "language"},
        {"essay",         "writing"},
        {"write",         "writing"},
        {"summarize",     "writing"},
        {"proofread",     "writing"},
        {"paragraph",     "writing"},

        // System & routing (always available on grim-text router)
        {"open",          "general"},
        {"close",         "general"},
        {"search",        "general"},
        {"find",          "general"},
        {"help",          "general"},

        // Creative
        {"story",         "creative"},
        {"poem",          "creative"},
        {"imagine",       "creative"},
        {"fiction",       "creative"},
        {"character",     "creative"},

        // Research & knowledge
        {"explain",       "knowledge"},
        {"define",        "knowledge"},
        {"history",       "knowledge"},
        {"compare",       "knowledge"},
        {"analyze",       "knowledge"},
    };
    return kw;
}

// =========================================================
// getSubjectTags
// =========================================================

std::vector<std::string> getSubjectTags(const std::string& raw_input) {
    if (raw_input.empty()) return {};

    // Lowercase the input
    std::string lower = raw_input;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Tokenize on whitespace and punctuation
    std::unordered_set<std::string> found_tags;
    std::string token;
    for (size_t i = 0; i <= lower.size(); ++i) {
        char c = (i < lower.size()) ? lower[i] : ' ';
        if (std::isalpha(static_cast<unsigned char>(c))) {
            token += c;
        } else {
            if (!token.empty()) {
                auto it = subjectKeywords().find(token);
                if (it != subjectKeywords().end()) {
                    found_tags.insert(it->second);
                }
                token.clear();
            }
        }
    }

    // Always include "general" if no specific tags found
    if (found_tags.empty()) {
        found_tags.insert("general");
    }

    return {found_tags.begin(), found_tags.end()};
}

} // namespace GRIM::MMO
