#include "ai.hpp"
#include "voice/voice.hpp"
#include "resources.hpp"
#include "commands/commands_core.hpp"
#include "error_manager.hpp"
#include "logger.hpp"
#include "personality_manager.hpp"
#include <cpr/cpr.h>
#include <fstream>
#include <sstream>
#include <future>

// ---------------- Globals ----------------
double g_silenceThreshold = 1e-6; // default, overridden in aiConfig
int g_silenceTimeoutMs    = 7000; // default 7 seconds
std::string g_whisperLanguage = "en";
int g_whisperMaxTokens        = 32;

// ====================================================
// Helpers: ensure voice section exists in memory
// ====================================================
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
            LOG_PHASE("Memory saved", true);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Memory", std::string("Failed to save memory.json: ") + e.what());
        LOG_PHASE("Memory save", false);
    }
}

void loadMemory() {
    std::ifstream f("memory.json");
    if (f) {
        try {
            f >> longTermMemory;
            LOG_PHASE("Memory loaded", true);
        } catch (const std::exception& e) {
            LOG_ERROR("Memory", std::string("Failed to parse memory.json: ") + e.what());
            longTermMemory = nlohmann::json::object();
            LOG_PHASE("Memory load", false);
        }
    } else {
        LOG_DEBUG("Memory", "No memory.json found. Creating new file.");
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
        std::string model   = aiConfig.value("default_model", "mistral");

        LOG_DEBUG("AI", "callAIAsync backend=" + backend + " model=" + model);

        try {
            if (backend == "ollama") {
                auto resp = cpr::Post(
                    cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/generate" },
                    cpr::Header{{"Content-Type","application/json"}},
                    cpr::Body{ nlohmann::json{{"model", model}, {"prompt", prompt}}.dump() }
                );
                if (resp.status_code == 200) {
                    auto j = nlohmann::json::parse(resp.text, nullptr, false);
                    return j.value("response", "");
                }
            }
            else if (backend == "localai" || backend == "openai") {
                std::string url =
                    (backend == "localai")
                        ? aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/chat/completions"
                        : "https://api.openai.com/v1/chat/completions";

                cpr::Header headers = {{"Content-Type","application/json"}};
                if (backend == "openai") {
                    auto apiKey = aiConfig["api_keys"].value("openai", "");
                    if (apiKey.empty()) return "[AI] Missing OpenAI API key";
                    headers["Authorization"] = "Bearer " + apiKey;
                }

                auto resp = cpr::Post(
                    cpr::Url{url},
                    headers,
                    cpr::Body{ nlohmann::json{
                        {"model", model},
                        {"messages", nlohmann::json::array({
                            {{"role","user"},{"content",prompt}}
                        })}
                    }.dump() }
                );
                if (resp.status_code == 200) {
                    auto j = nlohmann::json::parse(resp.text, nullptr, false);
                    if (j.contains("choices"))
                        return j["choices"][0]["message"]["content"].get<std::string>();
                }
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("AI", std::string("Exception: ") + e.what());
        }

        return "[AI] Backend call failed";
    });
}

// =========================================================
// Blocking AI call → returns CommandResult (with retry)
// =========================================================
CommandResult ai_process(const std::string& input) {
    CommandResult result;
    result.category  = "routine";
    result.color     = Colors::Cyan;
    result.success   = false;
    result.errorCode = "ERR_AI_BACKEND_UNAVAILABLE";

    const int maxRetries = 2;
    std::string reply;

    for (int attempt = 1; attempt <= maxRetries; ++attempt) {
        try {
            std::string prefix = GRIM::PersonalityManager::generatePrefix();
            auto future = callAIAsync(prefix + " " + input);
            reply = future.get();


            if (!reply.empty() && reply.rfind("[AI] Backend call failed", 0) != 0) {
                result.success = true;
                result.errorCode = "ERR_NONE";
                break;
            }

            LOG_DEBUG("AI", "Attempt " + std::to_string(attempt) + " failed: " + reply);
        }
        catch (const std::exception& e) {
            LOG_ERROR("AI", std::string("Exception on attempt ") + std::to_string(attempt) + ": " + e.what());
        }
    }

    // Memory update
    longTermMemory["last_input"] = input;
    longTermMemory["last_reply"] = reply;
    saveMemory();

    result.message = reply.empty() ? "[AI] Failed to process request" : reply;
    result.voice   = result.message;
    return result;
}
// =========================================================
// Unified AI interpreter — determines if input is a command or conversation
// =========================================================
CommandResult ai_interpret(const std::string& input, bool allowCommands)
{
    CommandResult result;
    result.category = "ai_interpret";
    result.color = Colors::Cyan;
    result.success = false;
    result.errorCode = "ERR_AI_INTERPRET_FAIL";

    try {
        // --- Build context prompt ---
        std::string personalityPrefix = GRIM::PersonalityManager::generatePrefix();

        std::string prompt =
            personalityPrefix +
            "You are GRIM, an AI companion and assistant. "
            "Analyze the user's input and decide what type of response it requires.\n"
            "Return JSON ONLY in the following format:\n"
            "{ \"intent\": \"command\" | \"conversation\" | \"question\", "
            "\"suggested_command\": string (if intent=command), "
            "\"response\": string (if intent=conversation or question) }\n\n"
            "Input: \"" + input + "\"\n"
            "Context (memory excerpt): " +
            longTermMemory.dump(1);

        // --- Query backend (Mistral / LocalAI / OpenAI etc.) ---
        std::string backend = resolveBackendURL();
        std::string model   = aiConfig.value("default_model", "mistral");

        LOG_DEBUG("AI", "ai_interpret backend=" + backend + " model=" + model);

        std::string reply;

        // Reuse the async infrastructure
        auto future = callAIAsync(prompt);
        reply = future.get();

        // --- Early validation of reply ---
        if (reply.empty() || reply.rfind("[AI] Backend call failed", 0) == 0) {
            LOG_ERROR("AI", "Backend returned error or empty response: " + reply);
            result.message = "[AI] Could not interpret input.";
            result.voice = "Sorry, I couldn't interpret that.";
            return result;
        }

        // --- Parse model response as JSON ---
        nlohmann::json j = nlohmann::json::parse(reply, nullptr, false);
        
        // ← FIX: Validate JSON before accessing it
        if (j.is_discarded() || !j.is_object()) {
            LOG_ERROR("AI", "Interpretation failed — non-JSON response: " + reply);
            result.message = "[AI] Could not interpret input.";
            result.voice = "Sorry, I couldn't interpret that.";
            return result;  // ← Early return to avoid accessing discarded JSON
        }

        // --- Route based on intent ---
        std::string intent = j.value("intent", "conversation");

        if (intent == "command" && allowCommands) {
            std::string suggested = j.value("suggested_command", "");
            if (!suggested.empty()) {
                LOG_DEBUG("AI", "Interpreter inferred command: " + suggested);
                result.message = suggested;
                result.category = "command_infer";
                result.success = true;
                result.errorCode = "ERR_NONE";
                return result;
            }
        }
        else if (intent == "conversation" || intent == "question") {
            std::string response = j.value("response", "");
            if (!response.empty()) {
                LOG_DEBUG("AI", "Interpreter conversational reply: " + response);
                result.message = response;
                result.voice = response;
                result.category = "conversation";
                result.success = true;
                result.errorCode = "ERR_NONE";

                // store context
                longTermMemory["last_conversation"] = {
                    {"input", input},
                    {"reply", response},
                    {"timestamp", std::time(nullptr)}
                };
                saveMemory();
                return result;
            }
        }

        // --- If all else fails ---
        result.message = "[AI] I couldn't determine your intent.";
        result.voice = "I couldn't determine your intent.";
    }
    catch (const std::exception& e) {
        LOG_ERROR("AI", std::string("Exception in ai_interpret: ") + e.what());
        result.message = "[AI] Interpretation error";
        result.voice = "Something went wrong while interpreting.";
    }

    return result;
}

// =========================================================
// Streaming / incremental AI call
// =========================================================
void ai_process_stream(
    const std::string& input,
    nlohmann::json& memory,
    const std::function<void(const std::string&)>& callback
) {
    std::string backend = resolveBackendURL();
    std::string model   = aiConfig.value("default_model", "mistral");

    LOG_DEBUG("AI", "ai_process_stream backend=" + backend + " model=" + model);

    bool success = false;

    try {
        if (backend == "ollama") {
            auto resp = cpr::Post(
                cpr::Url{ aiConfig.value("ollama_url", "http://127.0.0.1:11434") + "/api/generate" },
                cpr::Header{{"Content-Type","application/json"}},
                cpr::Body{ nlohmann::json{{"model", model}, {"prompt", input}, {"stream", true}}.dump() },
                cpr::Timeout{60000}
            );
            if (resp.status_code == 200) {
                std::istringstream ss(resp.text);
                std::string line;
                while (std::getline(ss, line)) {
                    if (!line.empty() && callback) callback(line + " ");
                }
                success = true;
            }
        }
        else if (backend == "localai" || backend == "openai") {
            std::string url =
                (backend == "localai")
                    ? aiConfig.value("localai_url","http://127.0.0.1:8080/v1") + "/chat/completions"
                    : "https://api.openai.com/v1/chat/completions";

            cpr::Header headers = {{"Content-Type","application/json"}};
            if (backend == "openai") {
                auto apiKey = aiConfig["api_keys"].value("openai", "");
                if (apiKey.empty()) {
                    if (callback) callback("[AI] Missing OpenAI API key\n");
                    LOG_ERROR("AI", "Missing OpenAI API key");
                    return;
                }
                headers["Authorization"] = "Bearer " + apiKey;
            }

            auto resp = cpr::Post(
                cpr::Url{url},
                headers,
                cpr::Body{ nlohmann::json{
                    {"model", model},
                    {"stream", true},
                    {"messages", nlohmann::json::array({
                        {{"role","user"},{"content",input}}
                    })}
                }.dump() },
                cpr::Timeout{60000}
            );

            if (resp.status_code == 200) {
                std::istringstream ss(resp.text);
                std::string line;
                while (std::getline(ss, line)) {
                    if (line.rfind("data:", 0) == 0) {
                        std::string chunk = line.substr(5);
                        if (chunk.find("[DONE]") != std::string::npos) break;
                        auto j = nlohmann::json::parse(chunk, nullptr, false);
                        if (!j.is_discarded() && j.contains("choices")) {
                            auto delta = j["choices"][0]["delta"];
                            if (delta.contains("content")) {
                                if (callback) callback(delta["content"].get<std::string>());
                            }
                        }
                    }
                }
                success = true;
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("AI", std::string("Exception in ai_process_stream: ") + e.what());
    }

    // Memory update
    memory["last_input"] = input;
    memory["last_reply"] = success ? "[streamed reply]" : "[AI] Stream failed";
}

// =========================================================
// Warmup
// =========================================================
void warmupAI() {
    LOG_DEBUG("AI", "Warming up...");
    auto f = callAIAsync("Hello");
    f.wait();
    LOG_PHASE("AI warmup complete", true);
}
