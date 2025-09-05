#include "ai.hpp"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>

// Global JSON object for long-term memory
nlohmann::json longTermMemory;

void loadMemory() {
    std::ifstream f("memory.json");
    if (f) {
        try {
            f >> longTermMemory;
            std::cerr << "[Memory] Loaded " << longTermMemory.size() << " entries\n";
        } catch (const std::exception& e) {
            std::cerr << "[Memory] Failed to parse memory.json: " << e.what() << "\n";
            longTermMemory = nlohmann::json::object();
        }
    } else {
        std::cerr << "[Memory] No memory.json found. Starting fresh.\n";
        longTermMemory = nlohmann::json::object();
    }
}

void saveMemory() {
    try {
        std::ofstream f("memory.json");
        if (f) {
            f << longTermMemory.dump(2);
            std::cerr << "[Memory] Saved " << longTermMemory.size() << " entries\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Memory] Failed to save memory.json: " << e.what() << "\n";
    }
}

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
        if (j.contains("response"))
            return j["response"].get<std::string>();
        return "[No 'response' field in Ollama reply]";
    } catch (const std::exception& e) {
        return std::string("Error parsing Ollama JSON: ") + e.what();
    }
}
