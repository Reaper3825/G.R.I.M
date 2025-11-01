#pragma once
#include <string>
#include <vector>
#include "web_result.hpp"

namespace GRIM {

struct WebSearchResult {
    std::string query;             // User query text
    std::string timestamp;         // UTC ISO time
    std::string engine;            // e.g. "DuckDuckGo", "Google"
    std::vector<WebResult> results;// Top N parsed + ranked results
    std::string summary;           // Summarized paragraph
    bool success = false;          // True if operation succeeded
    std::string error;             // Error message if any

    // Optional: for later use
    std::vector<float> queryEmbedding;
};
} // namespace GRIM
