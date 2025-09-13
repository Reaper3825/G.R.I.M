#include "ai.hpp"
#include "voice.hpp"

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
// AI + Voice configuration persistence
// =========================================================
void saveAIConfig(const std::string& filename) {
    try {
        std::ofstream f(filename);
        if (f) {
            f << aiConfig.dump(2);
            std::cerr << "[Config] Saved ai_config.json with "
                      << aiConfig.size() << " top-level keys\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Config] Failed to save ai_config.json: " << e.what() << "\n";
    }
}

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
            {"whisper_max_tokens", g_whisperMaxTokens},
            {"voice", {
                {"mode", "hybrid"},
                {"local_engine", "en_US-amy-medium.onnx"},
                {"cloud_engine", "openai"},
                {"rules", {
                    {"startup", "local"},
                    {"reminder", "local"},
                    {"summary", "cloud"},
                    {"banter", "cloud"}
                }},
                {"silence_threshold", g_silenceThreshold},
                {"silence_timeout_ms", g_silenceTimeoutMs},
                {"input_device_index", 0},
                {"tts_url", "http://127.0.0.1:8080/tts"}
            }},
            {"api_keys", {
                {"openai", "sk-your-openai-key-here"},
                {"elevenlabs", ""},
                {"azure", ""}
            }}
        };

        saveAIConfig(filename);
        return;
    }

    try {
        f >> aiConfig;
        std::cout << "[Config] AI config loaded successfully from " << filename << "\n";

        // ---- Global AI tuning ----
        g_silenceThreshold = aiConfig.value("silence_threshold", g_silenceThreshold);
        g_silenceTimeoutMs = aiConfig.value("silence_timeout_ms", g_silenceTimeoutMs);
        g_whisperLanguage  = aiConfig.value("whisper_language", g_whisperLanguage);
        g_whisperMaxTokens = aiConfig.value("whisper_max_tokens", g_whisperMaxTokens);

        // ---- Voice block ----
        if (aiConfig.contains("voice") && aiConfig["voice"].is_object()) {
            auto v = aiConfig["voice"];
            std::cout << "[Voice] Config loaded: mode="
                      << v.value("mode", "hybrid")
                      << " local=" << v.value("local_engine", "none")
                      << " cloud=" << v.value("cloud_engine", "none")
                      << " inputDevice=" << v.value("input_device_index", -1)
                      << "\n";

            g_silenceThreshold = v.value("silence_threshold", g_silenceThreshold);
            g_silenceTimeoutMs = v.value("silence_timeout_ms", g_silenceTimeoutMs);
        }

        saveAIConfig(filename);

    } catch (const std::exception& e) {
        std::cerr << "[Config] Error parsing " << filename << ": " << e.what() << std::endl;
    }
}

// =========================================================
// Backend resolver
// =========================================================
std::string resolveBackend() {
    std::string backend = aiConfig.value("backend", "auto");

    if (backend == "auto") {
        // Try Ollama first
        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("ollama_url","http://127.0.0.1:11434") + "/api/tags"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "ollama";
        } catch (...) {}

        // Try LocalAI
        try {
            auto r = cpr::Get(cpr::Url{aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/models"},
                              cpr::Timeout{1000});
            if (r.status_code == 200) return "localai";
        } catch (...) {}

        // Fallback
        return "openai";
    }

    return backend;
}

// =========================================================
// Core AI call
// =========================================================
std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() {
        std::string backend = resolveBackend();
        std::string model   = aiConfig.value("default_model", "mistral");

        std::cerr << "[AI] callAIAsync backend=" << backend << " model=" << model << "\n";

        if (backend == "ollama") {
            nlohmann::json body = {
                {"model", model},
                {"prompt", prompt},
                {"stream", false},
                {"keep_alive", "5m"}
            };

            std::string endpoint = aiConfig.value("ollama_url","http://127.0.0.1:11434") + "/api/generate";

            auto r = cpr::Post(cpr::Url{endpoint},
                               cpr::Header{{"Content-Type","application/json"}},
                               cpr::Body{body.dump()});
            if (r.status_code == 200) {
                try {
                    auto j = nlohmann::json::parse(r.text);
                    if (j.contains("response")) return j["response"].get<std::string>();
                    return r.text;
                } catch (...) { return r.text; }
            }
            return "Ollama error: " + r.text;
        }

        if (backend == "localai") {
            nlohmann::json body = {
                {"model", model},
                {"messages", {{{"role","user"},{"content",prompt}}}}
            };

            std::string endpoint = aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/chat/completions";

            auto r = cpr::Post(cpr::Url{endpoint},
                               cpr::Header{{"Content-Type","application/json"}},
                               cpr::Body{body.dump()});
            if (r.status_code == 200) {
                try {
                    auto j = nlohmann::json::parse(r.text);
                    if (j.contains("choices"))
                        return j["choices"][0]["message"]["content"].get<std::string>();
                    return r.text;
                } catch (...) { return r.text; }
            }
            return "LocalAI error: " + r.text;
        }

        if (backend == "openai") {
            nlohmann::json body = {
                {"model", model},
                {"messages", {{{"role","user"},{"content",prompt}}}}
            };

            auto r = cpr::Post(
                cpr::Url{"https://api.openai.com/v1/chat/completions"},
                cpr::Header{{"Content-Type","application/json"},
                            {"Authorization","Bearer " + aiConfig["api_keys"].value("openai","")}},
                cpr::Body{body.dump()}
            );
            if (r.status_code == 200) {
                try {
                    auto j = nlohmann::json::parse(r.text);
                    if (j.contains("choices"))
                        return j["choices"][0]["message"]["content"].get<std::string>();
                    return r.text;
                } catch (...) { return r.text; }
            }
            return "OpenAI error: " + r.text;
        }

        return std::string("[AI] No valid backend.");

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
            std::string chunk = word + " ";
            onChunk(chunk);

            // 🔹 Speak each chunk
            Voice::speakText(chunk, true);

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

        if (!reply.empty()) {
            Voice::speakText(reply, true);
        }

        return reply;
    } catch (const std::exception& e) {
        std::string err = std::string("[AI Error] ") + e.what();
        Voice::speakText(err, true);
        return err;
    }
}

// =========================================================
// Warm-up
// =========================================================
void warmupAI() {
    std::cout << "[AI] Warming up model...\n";
    auto warmupFuture = callAIAsync("Hello");
}
