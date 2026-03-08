// Multi-Model Orchestration (MMO) - Orchestrator
// Central dispatch: precompose → route → sub-model → synthesize → return.
//
// Flow:
//   1. Caller submits RequestContext (prompt + metadata + history).
//   2. Orchestrator ensures router is loaded (via ModelLoader).
//   3. Sends route request to router → gets RouteDecision (sub_model_id + composed_generation).
//   4. Ensures sub-model is loaded.
//   5. Sends composed_generation to sub-model (frozen brick) → gets raw sub-model output.
//   6. Feeds sub-model output back to router (synthesize) → gets final structured response.
//   7. Returns OrchestratorResult to caller.
//
// The orchestrator validates every boundary using Contracts (validateRequest,
// validateResponse) and delegates route-response parsing to ModelRouter.
//
// All resource admission goes through ModelLoader → ResourceCoordinator.
// The orchestrator never probes hardware directly.
//======================================================//
#pragma once

#include "Contracts.hpp"
#include "MemoryFacade.hpp"
#include "ModelLoader.hpp"
#include "ModelRegistry.hpp"
#include "RequestContext.hpp"
#include "../Router/ModelRouter.hpp"
#include "../Shared/MMD.hpp"
#include "../../ai/backends/IGenerationBackend.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Orchestrator configuration (from ai_config.json → mmo.orchestrator)
// =========================================================
struct OrchestratorConfig {
    int  route_timeout_ms           = 10000;
    int  generate_timeout_ms        = 30000;
    int  synthesize_timeout_ms      = 10000;
    int  max_submodels_per_request  = 1;    // v1: single sub-model
};

// =========================================================
// OrchestratorResult — output from generate()
// =========================================================
struct OrchestratorResult {
    bool        success       = false;
    std::string response;           // final synthesized text
    std::string request_id;
    std::string sub_model_id;       // which sub-model was dispatched
    std::string error;              // non-empty when !success
};

// =========================================================
// Orchestrator
//
// Usage:
//   Orchestrator orch(registry, loader, config);
//   RequestContext ctx;
//   ctx.request_id = generateId();
//   ctx.prompt = "Explain photosynthesis";
//   auto result = orch.generate(ctx);
//   if (result.success) { use(result.response); }
//
// Thread-safe: serializes generate() under its own mutex.
// =========================================================
class Orchestrator {
public:
    Orchestrator(ModelRegistry& registry,
                 ModelLoader& loader,
                 const OrchestratorConfig& config,
                 MemoryFacade* memory = nullptr);

    // Run the full orchestration flow for a request.
    // Returns Unavailable-style errors in OrchestratorResult::error
    // rather than throwing, so callers can handle gracefully.
    // Throws only on programming errors (null model, bad state).
    OrchestratorResult generate(const RequestContext& ctx);

    // Set or replace the memory facade (called from main after memory init).
    void setMemoryFacade(MemoryFacade* memory);

    // Register a backend for a specific model_id.
    // The orchestrator takes ownership of the backend.
    void registerBackend(const std::string& model_id,
                         std::unique_ptr<IGenerationBackend> backend);

    // Clean shutdown — unloads all models via loader.
    void shutdown();

private:
    // Build the router scope JSON from memory + caller metadata.
    // This is where memory retrieval enrichment happens.
    std::string buildRouterScope(const RequestContext& ctx) const;

    // Send a request to a model backend and get its response.
    // Dispatches through registered IGenerationBackend for the model.
    ResponseEnvelope callBackend(const ModelInfo& model,
                                const std::string& endpoint,
                                const RequestEnvelope& envelope,
                                int timeout_ms);

    // Convert an IGenerationBackend::GenerationResult back to a
    // ResponseEnvelope for contract validation.
    ResponseEnvelope resultToEnvelope(const GenerationResult& gen_result,
                                     const RequestEnvelope& envelope) const;

    ModelRegistry& registry_;
    ModelLoader&   loader_;
    MemoryFacade*  memory_;    // nullptr if memory not wired yet
    OrchestratorConfig config_;
    ModelRouter    router_;
    std::unordered_map<std::string, std::unique_ptr<IGenerationBackend>> backends_;
    mutable std::mutex mutex_;
};

} // namespace GRIM::MMO
