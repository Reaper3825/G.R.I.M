#pragma once
#include <string>
#include <vector>

namespace GRIM {

struct WebResult {
    std::string title;       // e.g. "Coqui XTTS 2.0 Launch"
    std::string url;         // e.g. "https://coqui.ai/blog/xtts2"
    std::string snippet;     // Short text or description
    float score = 0.0f;      // Semantic relevance score
    std::vector<float> embedding; // (optional) precomputed vector
};

} // namespace GRIM
