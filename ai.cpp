#include "ai.hpp"
#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <sstream>

// ---------------- Globals ----------------
nlohmann::json aiConfig;
nlohmann::json longTermMemory;

double g_silenceThreshold = 1e-6; // default, overridden in ai_config.json
int g_silenceTimeoutMs    = 7000; // default 7 seconds

// 🔹 Whisper tuning (configurable via ai_config.json)
std::string g_whisperLanguage = "en";
int g_whisperMaxTokens        = 32;

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
        }
    } else {
        std::cerr << "[Memory] No memory.json found. Creating new file.\n";
        longTermMemory = nlohmann::json::object();
    }

    if (!longTermMemory.contains("voice")) {
        longTermMemory["voice"] = {
            {"corrections", nlohmann::json::object()},
            {"shortcuts", nlohmann::json::object()},
            {"usage_counts", nlohmann::json::object()},
            {"last_command", ""}
        };
    }
    if (!longTermMemory.contains("voice_baseline")) {
        longTermMemory["voice_baseline"] = 0.0;
    }

    saveMemory();
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
// Voice helpers
// =========================================================
void rememberCorrection(const std::string& wrong, const std::string& right) {
    longTermMemory["voice"]["corrections"][wrong] = right;
    saveMemory();
}

void rememberShortcut(const std::string& phrase, const std::string& command) {
    longTermMemory["voice"]["shortcuts"][phrase] = command;
    saveMemory();
}

void incrementUsageCount(const std::string& command) {
    auto& counts = longTermMemory["voice"]["usage_counts"];
    if (!counts.contains(command)) counts[command] = 0;
    counts[command] = counts[command].get<int>() + 1;
    saveMemory();
}

void setLastCommand(const std::string& command) {
    longTermMemory["voice"]["last_command"] = command;
    saveMemory();
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
            {"silence_timeout_ms", g_silenceTimeoutMs},
            {"whisper_language", g_whisperLanguage},
            {"whisper_max_tokens", g_whisperMaxTokens}
        };
        std::ofstream out(filename);
        if (out) out << aiConfig.dump(2);
        return;
    }

    try {
        f >> aiConfig;
        std::cout << "[Config] AI config loaded successfully from " << filename << "\n";

        if (aiConfig.contains("silence_threshold"))
            g_silenceThreshold = aiConfig["silence_threshold"].get<double>();

        if (aiConfig.contains("silence_timeout_ms"))
            g_silenceTimeoutMs = aiConfig["silence_timeout_ms"].get<int>();

        if (aiConfig.contains("whisper_language"))
            g_whisperLanguage = aiConfig["whisper_language"].get<std::string>();

        if (aiConfig.contains("whisper_max_tokens"))
            g_whisperMaxTokens = aiConfig["whisper_max_tokens"].get<int>();

    } catch (const std::exception& e) {
        std::cerr << "[Config] Error parsing " << filename << ": " << e.what() << std::endl;
    }
}

// =========================================================
// Backend resolver
// =========================================================
std::string resolveBackendURL() {
    std::string backend = aiConfig.value("backend", "auto");

    if (backend == "auto") {
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

    if (backend == "ollama")
        return aiConfig.value("ollama_url", "http://127.0.0.1:11434");
    else
        return aiConfig.value("localai_url", "http://127.0.0.1:8080/v1");
}

// =========================================================
// Core AI call
// =========================================================
std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() {
        std::string url   = resolveBackendURL();
        std::string model = aiConfig.value("default_model", "mistral");

        nlohmann::json body = {
            {"model", model},
            {"prompt", prompt},
            {"stream", false},
            {"keep_alive", "5m"}
        };

        std::string endpoint;
        if (url.find("8080") != std::string::npos) {
            endpoint = url + "/chat/completions";
            body = {
                {"model", model},
                {"messages", {{{"role", "user"}, {"content", prompt}}}}
            };
        } else {
            endpoint = url + "/api/generate";
        }

        cpr::Response r = cpr::Post(
            cpr::Url{endpoint},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{body.dump()}
        );

        if (r.error) return std::string("Error calling AI: ") + r.error.message;
        if (r.status_code != 200) return "AI HTTP " + std::to_string(r.status_code) + ": " + r.text;

        try {
            auto j = nlohmann::json::parse(r.text);
            if (j.contains("response")) return j["response"].get<std::string>();
            if (j.contains("choices")) return j["choices"][0]["message"]["content"].get<std::string>();
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
        auto future = callAIAsync(input);
        std::string reply = future.get();

        std::istringstream iss(reply);
        std::string word;
        while (iss >> word) {
            onChunk(word + " ");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        memory["last_input"] = input;
        memory["last_reply"] = reply;
        saveMemory();
    } catch (const std::exception& e) {
        onChunk(std::string("[AI Stream Error] ") + e.what());
    }
}

// =========================================================
// Blocking AI call
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
// Warm-up
// =========================================================
void warmupAI() {
    std::cout << "[AI] Warming up model...\n";
    auto warmupFuture = callAIAsync("Hello");
}
