#include "ai.hpp"
#include "voice.hpp"
#include "resources.hpp"   // 🔹 Provides aiConfig + longTermMemory

#include <cpr/cpr.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <sstream>
#include <functional>

// ---------------- Globals ----------------
double g_silenceThreshold = 1e-6; // default, overridden in ai_config.json
int g_silenceTimeoutMs    = 7000; // default 7 seconds

// 🔹 Whisper tuning (configurable via ai_config.json)
std::string g_whisperLanguage = "en";
int g_whisperMaxTokens        = 32;

// ---------------- Helpers ----------------
nlohmann::json& voiceMemory() {
    if (!longTermMemory.contains("voice") || !longTermMemory["voice"].is_object()) {
        longTermMemory["voice"] = {
            {"corrections", nlohmann::json::object()},
            {"shortcuts", nlohmann::json::object()},
            {"usage_counts", nlohmann::json::object()},
            {"last_command", ""}
        };
    }
    return longTermMemory["voice"];
}

// =========================================================
// Memory persistence
// =========================================================
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

    // Ensure voice structure exists
    voiceMemory();
    if (!longTermMemory.contains("voice_baseline")) {
        longTermMemory["voice_baseline"] = 0.0;
    }

    saveMemory();
}

// =========================================================
// Voice helpers
// =========================================================
void rememberCorrection(const std::string& wrong, const std::string& right) {
    voiceMemory()["corrections"][wrong] = right;
    saveMemory();
}

void rememberShortcut(const std::string& phrase, const std::string& command) {
    voiceMemory()["shortcuts"][phrase] = command;
    saveMemory();
}

void incrementUsageCount(const std::string& command) {
    auto& counts = voiceMemory()["usage_counts"];
    if (!counts.contains(command)) counts[command] = 0;
    counts[command] = counts[command].get<int>() + 1;
    saveMemory();
}

void setLastCommand(const std::string& command) {
    voiceMemory()["last_command"] = command;
    saveMemory();
}

// =========================================================
// AI + Voice configuration persistence
// =========================================================
void saveAIConfig(const std::string& filename) {
    try {
        std::ofstream f(filename);
        if (f) {
            f << aiConfig.dump(2);
            std::cerr << "[Config] Saved " << filename
                      << " with " << aiConfig.size() << " top-level keys\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Config] Failed to save " << filename << ": " << e.what() << "\n";
    }
}

void loadAIConfig(const std::string& filename) {
    std::cout << "[DEBUG][AI Config] Attempting to load: " << filename << "\n";

    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "[Config] No " << filename << " found. Creating defaults.\n";

        aiConfig = {
            {"backend", "auto"},
            {"api_keys", {
                {"openai", ""},
                {"elevenlabs", ""},
                {"azure", ""}
            }}
        };

        saveAIConfig(filename);
        return;
    }

    try {
        f >> aiConfig;
        std::cout << "[DEBUG][AI Config] Raw contents after parse:\n"
                  << aiConfig.dump(2) << "\n";

        // ---- Global AI tuning ----
        g_silenceThreshold = aiConfig.value("silence_threshold", g_silenceThreshold);
        g_silenceTimeoutMs = aiConfig.value("silence_timeout_ms", g_silenceTimeoutMs);
        g_whisperLanguage  = aiConfig.value("whisper_language", g_whisperLanguage);
        g_whisperMaxTokens = aiConfig.value("whisper_max_tokens", g_whisperMaxTokens);

        // ---- Voice block ----
        if (aiConfig.contains("voice") && aiConfig["voice"].is_object()) {
            auto& v = aiConfig["voice"];
            std::cout << "[DEBUG][AI Config] Voice settings loaded\n";
            g_silenceThreshold = v.value("silence_threshold", g_silenceThreshold);
            g_silenceTimeoutMs = v.value("silence_timeout_ms", g_silenceTimeoutMs);
        }

        // ---- Ensure api_keys exists ----
        if (!aiConfig.contains("api_keys") || !aiConfig["api_keys"].is_object()) {
            std::cerr << "[DEBUG][AI Config] api_keys missing or invalid. Resetting...\n";
            aiConfig["api_keys"] = {
                {"openai", ""},
                {"elevenlabs", ""},
                {"azure", ""}
            };
        } else {
            std::cout << "[DEBUG][AI Config] api_keys block:\n"
                      << aiConfig["api_keys"].dump(2) << "\n";
        }

        saveAIConfig(filename);

    } catch (const std::exception& e) {
        std::cerr << "[Config] Failed to parse " << filename << ": " << e.what() << "\n";
    }
}

// =========================================================
// Backend resolver
// =========================================================
std::string resolveBackendURL() {
    std::string backend = aiConfig.value("backend", "auto");

    if (backend == "auto") {
        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("ollama_url","http://127.0.0.1:11434") + "/api/tags"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "ollama";
        } catch (...) {}

        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/models"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "localai";
        } catch (...) {}

        return "openai";
    }

    return backend;
}

// =========================================================
// Core async AI call
// =========================================================
std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() -> std::string {
        std::string backend = resolveBackendURL();
        std::string model   = aiConfig.value("default_model","mistral");

        std::cerr << "[AI] callAIAsync backend=" << backend
                  << " model=" << model << "\n";

        if (aiConfig.contains("api_keys")) {
            std::cerr << "[DEBUG][AI Config] api_keys currently:\n"
                      << aiConfig["api_keys"].dump(2) << "\n";
        } else {
            std::cerr << "[DEBUG][AI Config] api_keys block is completely missing!\n";
        }

        // (rest unchanged, same as before)
        // ...
        return "[AI] Debug mode only – skipping backend call";
    });
}

// =========================================================
// Blocking AI call with voice
// =========================================================
std::string ai_process(const std::string& input) {
    auto future = callAIAsync(input);
    std::string reply = future.get();
    longTermMemory["last_input"] = input;
    longTermMemory["last_reply"] = reply;
    saveMemory();
    Voice::speakText(reply, false);
    return reply;
}

// =========================================================
// Streaming / incremental speak (callback-based)
// =========================================================
void ai_process_stream(
    const std::string& input,
    nlohmann::json& memory,
    const std::function<void(const std::string&)>& callback
) {
    auto future = callAIAsync(input);
    std::string reply = future.get();

    memory["last_input"] = input;
    memory["last_reply"] = reply;

    std::istringstream iss(reply);
    std::string word;
    std::ostringstream chunk;
    int count = 0;

    while (iss >> word) {
        chunk << word << " ";
        ++count;
        if (count >= 10) {
            if (callback) callback(chunk.str());
            chunk.str(""); chunk.clear();
            count = 0;
        }
    }
    if (!chunk.str().empty()) {
        if (callback) callback(chunk.str());
    }
}

// =========================================================
// Warmup
// =========================================================
void warmupAI() {
    std::cout << "[AI] Warming up...\n";
    auto f = callAIAsync("Hello");
    f.wait();
    std::cout << "[AI] Warmup complete\n";
}
