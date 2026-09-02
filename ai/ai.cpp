#include "ai.hpp"
#include "bootstrap/bootstrap.hpp"  // MMO orchestrator globals
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "nlp/NlpAnnotation.hpp"
#include "nlp/RouterMetadataBuilder.hpp"
#include "memory/atomic_writer.hpp"
#include "resources.hpp"
#include "settings/intervention_gate.hpp"
#include "logger.hpp"
#include "personality_manager.hpp"
#include "location.hpp"  // For location context
#include "grim_text_server_manager.hpp"
#include "../MMO/Backends/GrimNativeBackend.hpp"
#include "../MMO/Backends/OllamaBackend.hpp"
#include <fstream>
#include <future>
#include <algorithm>
#include <utility>

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
        GRIM::AtomicWriter::writeString("memory.json", longTermMemory.dump(2));
        LOG_PHASE("Memory saved", true);
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
// Core async AI call -- all requests go through MMO orchestrator
// =========================================================
static constexpr const char* kDefaultSessionId = "default";

void clearConversationHistory() {
    auto& scm = GRIM::MMO::SessionContextManager::instance();
    scm.clearHistory(kDefaultSessionId);
    LOG_DEBUG("AI", "Conversation history cleared");
}

// =========================================================
// callOllamaDirect -- bypass orchestrator for Ollama backend
// =========================================================
static std::vector<GRIM::MMO::HistoryEntry> buildDirectHistory(
    const std::string& session_id) {
    auto& scm = GRIM::MMO::SessionContextManager::instance();
    auto messages = scm.getMessages(session_id);

    std::vector<GRIM::MMO::HistoryEntry> history;
    history.reserve(messages.size());
    for (const auto& msg : messages) {
        history.push_back({msg.role, msg.content});
    }

    return history;
}

static bool isGrimTextModelName(const std::string& model_name) {
    std::string normalized = model_name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized == "grim-text"
        || normalized == "grim_text"
        || normalized == "grim-text-router";
}

static std::string formatMMORouteState() {
    bool mmo_config_enabled = false;
    if (aiConfig.contains("mmo") && aiConfig["mmo"].is_object()) {
        mmo_config_enabled = aiConfig["mmo"].value("enabled", false);
    }

    bool mmo_registry_enabled = GRIM::MMO::ModelRegistry::instance().isEnabled();
    return "mmo_config_enabled=" + std::string(mmo_config_enabled ? "true" : "false")
        + ", mmo_registry_enabled=" + std::string(mmo_registry_enabled ? "true" : "false")
        + ", orchestrator=" + std::string(g_orchestrator ? "ready" : "null");
}

static int resolveGrimTextRequestTimeoutMs() {
    constexpr int kDefaultGrimTextTimeoutMs = 600000;

    if (!aiConfig.contains("grim_text_timeout_ms")) {
        return kDefaultGrimTextTimeoutMs;
    }

    const int timeout_ms = aiConfig.value("grim_text_timeout_ms", kDefaultGrimTextTimeoutMs);
    if (timeout_ms <= 0) {
        throw std::runtime_error("ai_config grim_text_timeout_ms must be > 0");
    }

    return timeout_ms;
}

static void ensureGrimTextServerReady(const std::string& server_url) {
    auto& server_manager = GRIM::GRIMTextServerManager::getInstance();
    server_manager.setServerURL(server_url);

    if (server_manager.checkHealth(2000)) {
        LOG_DEBUG("AI", "GRIM-text server already healthy at " + server_url);
        return;
    }

    LOG_DEBUG("AI", "Active model targets GRIM-text; starting GRIM-text server at " + server_url);
    if (!GRIM::startGRIMTextServer()) {
        throw std::runtime_error(
            "GRIM-text server is required for active model '"
            + aiConfig.value("default_model", std::string("grim-text"))
            + "' but startup failed");
    }

    if (!server_manager.checkHealth(2000)) {
        throw std::runtime_error(
            "GRIM-text server startup completed but health check still failed at " + server_url);
    }
}

static std::string callOllamaDirect(
    const std::string& prompt,
    const std::string& session_id) {
    std::string url   = aiConfig.value("ollama_url", "http://127.0.0.1:11434");
    std::string model = aiConfig.value("default_model", "llama3.1:8b");

    GRIM::MMO::OllamaBackend backend(url, "ollama-direct", model);

    std::vector<GRIM::MMO::HistoryEntry> history = buildDirectHistory(session_id);

    GRIM::MMO::GenerationOptions opts;
    opts.timeout_ms = 60000;

    GRIM::MMO::GenerationResult gen = backend.generateWithHistory(prompt, history, opts);
    if (gen.success) {
        return gen.text;
    }

    LOG_ERROR("AI", "Ollama direct call failed: " + gen.error);
    return "[AI] Backend call failed: " + gen.error;
}

static std::string callGrimTextDirect(
    const std::string& prompt,
    const std::string& session_id) {
    std::string url = aiConfig.value("grim_text_url", "http://127.0.0.1:11435");
    ensureGrimTextServerReady(url);

    GRIM::MMO::GrimNativeBackend backend(url, "grim-text-direct");
    std::vector<GRIM::MMO::HistoryEntry> history = buildDirectHistory(session_id);

    GRIM::MMO::GenerationOptions opts;
    opts.timeout_ms = resolveGrimTextRequestTimeoutMs();

    GRIM::MMO::GenerationResult gen = backend.generateWithHistory(prompt, history, opts);
    if (gen.success) {
        return gen.text;
    }

    LOG_ERROR("AI", "GRIM-text direct call failed: " + gen.error);
    return "[AI] Backend call failed: " + gen.error;
}

static std::string callActiveModelDirect(
    const std::string& prompt,
    const std::string& session_id) {
    std::string active_model = aiConfig.value("default_model", "llama3.1:8b");
    LOG_DEBUG("AI", "Direct model path hit: active_model=" + active_model + ", " + formatMMORouteState());
    if (isGrimTextModelName(active_model)) {
        return callGrimTextDirect(prompt, session_id);
    }
    return callOllamaDirect(prompt, session_id);
}

std::future<std::string> callAIAsync(const std::string& prompt) {
    return callAIAsync(prompt, kDefaultSessionId);
}

std::future<std::string> callAIAsync(
    const std::string& prompt,
    const std::string& session_id) {
    return std::async(std::launch::async, [prompt, session_id]() -> std::string {
        // Check aiConfig["backend"] to decide routing
        std::string backend = aiConfig.value("backend", "auto");



        if (backend == "grim_native") {
        LOG_DEBUG("AI", "backend=grim_native direct dispatch selected; " + formatMMORouteState());
        return callGrimTextDirect(prompt, session_id);
        }



        // Direct Ollama path -- no orchestrator needed
        if (backend == "ollama") {
            LOG_DEBUG("AI", "backend=ollama direct dispatch selected; " + formatMMORouteState());
            return callActiveModelDirect(prompt, session_id);
        }



        // All other backends route through MMO orchestrator
        if (!g_orchestrator) {
            throw std::runtime_error(
                "g_orchestrator is NULL -- MMO must be initialized before calling AI. "
                "Check bootstrap/bootstrap.cpp bootstrapMMOLayer().");
        }

        auto& registry = GRIM::MMO::ModelRegistry::instance();
        if (!registry.isEnabled()) {
            LOG_DEBUG("AI", "MMO toggled off -- falling back to direct model dispatch; " + formatMMORouteState());
            return callOllamaDirect(prompt, session_id);
        }

        // Produce NlpAnnotation for structured routing
        auto& scm = GRIM::MMO::SessionContextManager::instance();
        GRIM::MMO::ContextSnapshotV2 snapshot = scm.snapshot(session_id);
        GRIM::NlpAnnotation annotation;
        if (Settings::interventionEnabled("nlp_annotation")) {
            annotation = GRIM::annotate(prompt, snapshot);
        } else {
            // Intent system gate -- skip NLP annotation; hand the router a bare
            // payload so the model drives routing without local enrichment.
            annotation.raw_text = prompt;
            annotation.normalized_text = prompt;
        }

        // Build router metadata envelope
        auto& toolReg = GRIM::MMO::ToolRegistry::instance();
        GRIM::RouterMetadataBuilder builder;
        builder.setAnnotation(annotation)
               .setContextV2(snapshot);
        GRIM::RouterMetadata meta = builder.build();

        GRIM::MMO::RequestContext ctx;
        ctx.request_id = std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
        ctx.session_id = session_id;
        ctx.turn_id = ctx.request_id;
        ctx.prompt = prompt;
        ctx.metadata_json = meta.toJson().dump();
        ctx.tool_registry_version = toolReg.version();

        // Carry the same prior conversation used by direct backends into MMO
        // requests. The current prompt is intentionally not in this history.
        for (const auto& message : scm.getMessages(session_id)) {
            if (message.role == "system") {
                ctx.system_prompt = message.content;
                continue;
            }

            GRIM::MMO::TurnSummary turn;
            turn.role = message.role;
            turn.text = message.content;
            ctx.recent_turns.push_back(std::move(turn));
        }

        auto result = g_orchestrator->generate(ctx);
        if (result.success) {
            return result.response;
        }

        LOG_ERROR("AI", "MMO orchestrator failed: " + result.error);
        return "[AI] Orchestrator error: " + result.error;
    });
}

// =========================================================
// Blocking AI call -> returns CommandResult (with retry)
// =========================================================
CommandResult ai_process(const std::string& input) {
    return ai_process(input, kDefaultSessionId);
}

CommandResult ai_process(
    const std::string& input,
    const std::string& session_id) {
    CommandResult result;
    result.category  = "routine";
    result.color     = Colors::Cyan;
    result.success   = false;
    result.errorCode = "ERR_AI_BACKEND_UNAVAILABLE";

    const int maxRetries = 2;
    std::string reply;
    auto& session = GRIM::MMO::SessionContextManager::instance();

    if (aiConfig.contains("personality")) {
        const auto& personality = aiConfig["personality"];
        if (personality.value("use_custom_prompt", false)) {
            session.setSystemPrompt(
                session_id,
                personality.value("custom_prompt", "You are GRIM. Be brief and direct."));
        }
    }

    const size_t maxHistory = aiConfig.value("conversation_history_size", 10);
    // Only prior turns belong in the history for this request. Recording the
    // current input before inference would duplicate it beside the prompt.
    session.trimHistory(session_id, maxHistory);

    for (int attempt = 1; attempt <= maxRetries; ++attempt) {
        try {
            std::string prefix = Settings::interventionEnabled("personality")
                ? GRIM::PersonalityManager::generatePrefix() : "";

            // Add location context for location-aware conversations
            std::string locationContext = "";
            std::string lowerInput = input;
            std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
            
            bool isLocationRelevant = (
                lowerInput.find("weather") != std::string::npos ||
                lowerInput.find("near me") != std::string::npos ||
                lowerInput.find("nearby") != std::string::npos ||
                lowerInput.find("local") != std::string::npos ||
                lowerInput.find("where am i") != std::string::npos ||
                lowerInput.find("my location") != std::string::npos ||
                lowerInput.find("around here") != std::string::npos
            );
            
            if (isLocationRelevant && (g_location.lat != 0.0 || g_location.lon != 0.0)) {
                locationContext = " [USER LOCATION: " + g_location.fullAddress() + "]";
            }
            
            auto future = callAIAsync(
                prefix + locationContext + " " + input,
                session_id);
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

    // ai_process is the owner of raw conversational history. Persist exactly
    // one user turn and, when inference succeeded, its assistant response.
    session.addMessage(session_id, "user", input);
    if (result.success) {
        session.addMessage(session_id, "assistant", result.message);
    }
    session.trimHistory(session_id, maxHistory);

    return result;
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
