#include "ai.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

// ---------------- Globals ----------------
nlohmann::json longTermMemory;
nlohmann::json aiConfig; // backend, endpoints, style

// =========================================================
// Memory persistence
// =========================================================
void loadMemory() {
    std::ifstream f("memory.json");
    if (f) {
        try {
            f >> longTermMemory;
            std::cerr << "[Memory] Loaded " << longTermMemory.size() << " entries\n";
        } catch (const std::exception& e) {
            std::cerr << "[Memory] Failed to parse memory.json: " << e.what() << "\n";
            longTermMemory = nlohmann::json::object();
            saveMemory();
        }
    } else {
        std::cerr << "[Memory] No memory.json found. Creating new file.\n";
        longTermMemory = nlohmann::json::object();
        saveMemory();
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

// =========================================================
// AI configuration
// =========================================================
void loadAIConfig(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Config] No " << filename << " found, creating defaults.\n";
        aiConfig = {
            {"backend", "auto"},
            {"ollama_url", "http://127.0.0.1:11434"},
            {"localai_url", "http://127.0.0.1:8080/v1"},
            {"default_model", "mistral"} // fallback model
        };
        std::ofstream out(filename);
        if (out) {
            out << aiConfig.dump(2);
            std::cerr << "[Config] Default " << filename << " created.\n";
        }
        return;
    }

    try {
        f >> aiConfig;
        std::cout << "[Config] AI config loaded successfully from " << filename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Config] Error parsing " << filename << ": " << e.what() << std::endl;
        aiConfig = {
            {"backend", "auto"},
            {"ollama_url", "http://127.0.0.1:11434"},
            {"localai_url", "http://127.0.0.1:8080/v1"},
            {"default_model", "mistral"}
        };
    }
}

// =========================================================
// Backend resolver
// =========================================================
std::string resolveBackendURL() {
    std::string backend = aiConfig.value("backend", "auto");

    if (backend == "auto") {
        // Pick defaults by platform
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

    std::string url;
    if (backend == "ollama") {
        url = aiConfig.value("ollama_url", "http://127.0.0.1:11434");
    } else {
        url = aiConfig.value("localai_url", "http://127.0.0.1:8080/v1");
    }

    std::cout << "[Config] Using backend: " << backend << " (" << url << ")\n";
    return url;
}

// =========================================================
// Core AI call
// =========================================================
std::string callAI(const std::string& prompt) {
    std::string url = resolveBackendURL();
    std::string model = aiConfig.value("default_model", "mistral");

    nlohmann::json body = {
        {"model", model},
        {"prompt", prompt},
        {"stream", false}
    };

    // Choose endpoint format depending on backend
    std::string endpoint;
    if (url.find("8080") != std::string::npos) {
        // LocalAI → OpenAI-compatible
        endpoint = url + "/chat/completions";
        body = {
            {"model", model},
            {"messages", {{{"role", "user"}, {"content", prompt}}}}
        };
    } else {
        // Ollama
        endpoint = url + "/api/generate";
    }

    std::cout << "[AI] Sending prompt to " << endpoint << " using model " << model << "\n";

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
        return "[AI] No valid field in response.";
    } catch (const std::exception& e) {
        return std::string("Error parsing AI JSON: ") + e.what();
    }
}
