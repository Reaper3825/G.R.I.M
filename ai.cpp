#include "ai.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>

// ---------------- Globals ----------------
extern nlohmann::json aiConfig;
extern nlohmann::json longTermMemory;
double g_silenceThreshold = 1e-6; // default, can be overridden in ai_config.json
int g_silenceTimeoutMs = 7000; // default 7 seconds



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
            {"default_model", "mistral"},
            {"silence_threshold", g_silenceThreshold},
            {"silence_timeout_ms", g_silenceTimeoutMs}
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

        // ✅ Load silence threshold if present
        if (aiConfig.contains("silence_threshold")) {
            g_silenceThreshold = aiConfig["silence_threshold"].get<double>();
            std::cout << "[Voice] Silence threshold set to " << g_silenceThreshold << "\n";
        }

        // ✅ Load silence timeout if present
        if (aiConfig.contains("silence_timeout_ms")) {
            g_silenceTimeoutMs = aiConfig["silence_timeout_ms"].get<int>();
            std::cout << "[Voice] Silence timeout set to " 
                      << g_silenceTimeoutMs << " ms\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[Config] Error parsing " << filename << ": " << e.what() << std::endl;
        aiConfig = {
            {"backend", "auto"},
            {"ollama_url", "http://127.0.0.1:11434"},
            {"localai_url", "http://127.0.0.1:8080/v1"},
            {"default_model", "mistral"},
            {"silence_threshold", g_silenceThreshold},
            {"silence_timeout_ms", g_silenceTimeoutMs}
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
// Core AI call (async)
// =========================================================
std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() {
        std::string url = resolveBackendURL();
        std::string model = aiConfig.value("default_model", "mistral");

        nlohmann::json body = {
            {"model", model},
            {"prompt", prompt},
            {"stream", false},
            {"keep_alive", "5m"}
        };

        std::string endpoint;
        if (url.find("8080") != std::string::npos) {
            // LocalAI (OpenAI-compatible)
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
            return std::string("[AI] No valid field in response.");
        } catch (const std::exception& e) {
            return std::string("Error parsing AI JSON: ") + e.what();
        }
    });
}


// =========================================================
// Streaming AI call
// =========================================================
void ai_process_stream(const std::string& input, nlohmann::json& memory,
                       const std::function<void(const std::string&)>& onChunk) {
    try {
        std::string url = resolveBackendURL();
        std::string model = aiConfig.value("default_model", "mistral");

        // NOTE: streaming via CPR is limited — this example simulates chunking
        auto future = callAIAsync(input);
        std::string reply = future.get();

        // Simulate stream by chunking reply into words
        std::istringstream iss(reply);
        std::string word;
        while (iss >> word) {
            onChunk(word + " ");
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // pacing
        }

        // Save to memory
        memory["last_input"] = input;
        memory["last_reply"] = reply;
        saveMemory();

    } catch (const std::exception& e) {
        onChunk(std::string("[AI Stream Error] ") + e.what());
    }
}


// =========================================================
// Blocking AI call (simple wrapper)
// =========================================================
std::string ai_process(const std::string& input, nlohmann::json& memory) {
    try {
        auto future = callAIAsync(input);
        std::string reply = future.get();

        memory["last_input"] = input;
        memory["last_reply"] = reply;
        saveMemory();

        return reply;
    } catch (const std::exception& e) {
        return std::string("[AI Error] ") + e.what();
    }
}


// =========================================================
// Warm-up on launch
// =========================================================
void warmupAI() {
    std::cout << "[AI] Warming up model...\n";
    auto warmupFuture = callAIAsync("Hello");
    // Let it run in background
}
