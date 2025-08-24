// ai.cpp
#include "ai.hpp"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <iostream>

std::string callAI(const std::string& prompt) {
    nlohmann::json body = {
        {"model", "mistral"},
        {"prompt", prompt},
        {"stream", false}
    };

    cpr::Response r = cpr::Post(
        cpr::Url{"http://localhost:11434/api/generate"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()}
    );

    // Network error (connection refused, etc.)
    if (r.error) {
        return std::string("Error calling Ollama: ") + r.error.message;
    }
    // Non-200 HTTP
    if (r.status_code != 200) {
        return "Ollama HTTP " + std::to_string(r.status_code) + ": " + r.text;
    }

    try {
        auto j = nlohmann::json::parse(r.text);
        if (j.contains("response")) return j["response"].get<std::string>();
        return "[No 'response' field in Ollama reply]";
    } catch (const std::exception& e) {
        return std::string("Error parsing Ollama JSON: ") + e.what();
    }
}
