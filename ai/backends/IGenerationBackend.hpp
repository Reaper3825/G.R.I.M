// IGenerationBackend — abstract backend interface for model inference
//
// Every model the orchestrator can call (router, sub-model, etc.)
// is accessed through this interface.  Implementations handle
// transport (HTTP, in-process, pipe, etc.) and serialization;
// the orchestrator only sees `generate()` / `isAvailable()`.
//
// Implementations:
//   GrimNativeRouterBackend   — HTTP to grim_text_server.exe
//   OllamaBackend             — Ollama API
//   LlamaCppBackend           — llama.cpp server
//   ExternalBackend            — arbitrary HTTP endpoint
//
// All backends are READ-ONLY from the model's perspective:
//   - No weight writes through this path.
//   - LoRA loading is a ModelLoader concern, not a backend concern.
//======================================================//
#pragma once
#include "DataCollection/reasoning_state.hpp"
#include <optional>

#include "../../MMO/Shared/MMD.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// GenerationOptions — per-call inference parameters
//
// Callers fill this with the body-built metadata before
// passing to generate(). Sub-model backends may ignore
// fields that don't apply (e.g. metadata_json on frozen bricks).
// =========================================================
struct GenerationOptions {
    int         max_tokens           = 0;   // 0 = use model default
    float       temperature          = 0.0f; // 0 = use model default
    float       top_p                = 0.0f; // 0 = use model default
    int         top_k                = 0;
    int64_t     seed                 = -1;  // -1 = backend default/random

    // Body-built structured metadata (JSON).
    // Router calls include the full precomposed metadata.
    // Sub-model calls include ONLY the composed_generation.
    std::string metadata_json;

    // Tool-surface summary for the router to consider.
    // Empty for sub-model calls.
    std::string tool_summary;

    // Timeout for this specific call (ms).  0 = no timeout.
    int         timeout_ms           = 0;

    // MMO envelope transport — when populated, the backend
    // should POST the envelope JSON directly to the MMO
    // endpoint instead of constructing its own request body.
    // Populated by Orchestrator::callBackend.
    std::string envelope_json;       // Full RequestEnvelope as JSON
    std::string mmo_endpoint;        // e.g. "/api/mmo/route"

    // Native chat/generate state envelope, separate from text/history.
    std::optional<GRIM::ReasoningState> reasoning_state;
};

// =========================================================
// HistoryEntry — single conversation turn for multi-turn
// =========================================================
struct HistoryEntry {
    std::string role;    // "user" | "assistant" | "system"
    std::string content;
};

// =========================================================
// GenerationResult — what the backend returns
// =========================================================
struct GenerationResult {
    bool        success   = false;
    std::string text;              // generated text when success
    std::string error;             // error message when !success
    int         tokens_used = 0;   // tokens consumed (if reported)
};

// =========================================================
// IGenerationBackend
//
// Thread-safety: implementations MUST be safe for concurrent
// calls to generate() from different requests.  isAvailable()
// and getBackendId() must be safe to call at any time.
// =========================================================
class IGenerationBackend {
public:
    virtual ~IGenerationBackend() = default;

    // ── Core generation ───────────────────────────────────

    // Single-turn generation.
    virtual GenerationResult generate(
        const std::string& prompt,
        const GenerationOptions& options) = 0;

    // Multi-turn generation with conversation history.
    virtual GenerationResult generateWithHistory(
        const std::string& prompt,
        const std::vector<HistoryEntry>& history,
        const GenerationOptions& options) = 0;

    // ── Lifecycle ─────────────────────────────────────────

    // True if the backend is reachable and ready for requests.
    virtual bool isAvailable() const = 0;

    // Stable identifier for this backend instance.
    // Format: "{backend_type}:{model_id}" e.g. "grim_text:router"
    virtual std::string getBackendId() const = 0;

    // The BackendType this implementation handles.
    virtual BackendType getBackendType() const = 0;
};

} // namespace GRIM::MMO
