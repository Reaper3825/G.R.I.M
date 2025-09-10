#include "ai.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

// Global JSON objects
nlohmann::json longTermMemory;
nlohmann::json aiConfig; // backend + tone/style

// ---------------- Memory persistence ----------------
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

// ---------------- AI configuration ----------------
void loadAIConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Config] No " << filename << " found, using defaults.\n";
        aiConfig = nlohmann::json::object();
        return;
    }

    try {
        f >> aiConfig;
        std::cout << "[Config] AI config loaded successfully from " << filename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Config] Error parsing " << filename << ": " << e.what() << std::endl;
        aiConfig = nlohmann::json::object();
    }
}

// ---------------- Backend resolver ----------------
std::string resolveBackendURL() {
    std::string backend = "auto";
    if (aiConfig.contains("backend")) {
        backend = aiConfig["backend"].get<std::string>();
    }

    if (backend == "auto") {
        // Detect OS
#if defined(_WIN32) || defined(_WIN64)
        backend = "ollama";
#elif defined(__linux__)
        backend = "localai";
#elif defined(__APPLE__)
        backend = "ollama";
#else
        backend = "localai";
#endif
    }

    if (backend == "ollama") {
        return aiConfig.value("ollama_url", "http://127.0.0.1:11434");
    } else {
        return aiConfig.value("localai_url", "http://127.0.0.1:8080/v1");
    }
}

// ---------------- Core AI call ----------------
std::string callAI(const std::string& prompt) {
    std::string url = resolveBackendURL();

    nlohmann::json body = {
        {"model", "mistral"},   // TODO: configurable
        {"prompt", prompt},
        {"stream", false}
    };

    // Build correct endpoint depending on backend type
    std::string endpoint;
    if (url.find("8080") != std::string::npos) {
        // LocalAI expects /chat/completions
        endpoint = url + "/chat/completions";
        body = {
            {"model", "gemma:2b"},
            {"messages", {{{"role", "user"}, {"content", prompt}}}}
        };
    } else {
        // Ollama
        endpoint = url + "/api/generate";
    }

    cpr::Response r = cpr::Post(
        cpr::Url{endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()}
    );

    if (r.error) {
        return std::string("Error calling AI: ") + r.error.message;
    }
    if (r.status_code != 200) {
        return "AI HTTP " + std::to_string(r.status_code) + ": " + r.text;
    }

    try {
        auto j = nlohmann::json::parse(r.text);
        if (j.contains("response")) {
            return j["response"].get<std::string>(); // Ollama
        } else if (j.contains("choices")) {
            return j["choices"][0]["message"]["content"].get<std::string>(); // LocalAI
        }
        return "[No valid field in AI reply]";
    } catch (const std::exception& e) {
        return std::string("Error parsing AI JSON: ") + e.what();
    }
}
