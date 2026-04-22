// Multi-Model Orchestration (MMO) - Orchestrator
// See Orchestrator.hpp for interface documentation.
//
// Migration: envelope builders → Contracts, route parsing → ModelRouter,
// correlation validation → Contracts::validateResponse.
//======================================================//
#include "Orchestrator.hpp"
#include "ToolRegistry.hpp"
#include "CorrectionTuple.hpp"
#include "ToolTrainingParser.hpp"

#include "../../logger.hpp"
#include "../../memory/memory_buffer_rotation.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

Orchestrator::Orchestrator(ModelRegistry& registry,
                           ModelLoader& loader,
                           const OrchestratorConfig& config,
                           MemoryFacade* memory)
    : registry_(registry)
    , loader_(loader)
    , memory_(memory)
    , config_(config)
    , tool_gap_planner_(ToolRegistry::instance()) {}

// =========================================================
// setMemoryFacade — late-bind memory after init
// =========================================================

void Orchestrator::setMemoryFacade(MemoryFacade* memory) {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_ = memory;
}

// =========================================================
// registerBackend — register an IGenerationBackend for a model
// =========================================================

void Orchestrator::registerBackend(const std::string& model_id,
                                    std::unique_ptr<IGenerationBackend> backend) {
    if (model_id.empty()) {
        throw std::runtime_error(
            "Orchestrator::registerBackend: model_id is empty");
    }
    if (!backend) {
        throw std::runtime_error(
            "Orchestrator::registerBackend: backend is null for '"
            + model_id + "'");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    backends_[model_id] = std::move(backend);
    LOG_DEBUG("MMO_ORCH", "Registered backend for model '" + model_id + "'");
}

// =========================================================
// generate — full orchestration flow
// =========================================================

OrchestratorResult Orchestrator::generate(const RequestContext& ctx) {
    LOG_DEBUG("MMO_ORCH", "[TRACE] generate() waiting for mutex (req=" + ctx.request_id + ")");
    std::lock_guard<std::mutex> lock(mutex_);
    LOG_DEBUG("MMO_ORCH", "[TRACE] generate() acquired mutex (req=" + ctx.request_id + ")");

    OrchestratorResult result;
    result.request_id = ctx.request_id;

    // ── Step 1: Get router model info ──
    const ModelInfo* router_info = registry_.getRouter();
    if (!router_info) {
        throw std::runtime_error(
            "Orchestrator::generate: no router model registered");
    }

    // ── Step 2: Ensure router is loaded ──
    LOG_DEBUG("MMO_ORCH", "[TRACE] ensureLoaded('" + router_info->id + "')");
    LoadResult load_result = loader_.ensureLoaded(router_info->id);
    LOG_DEBUG("MMO_ORCH", "[TRACE] ensureLoaded returned " + std::to_string(static_cast<int>(load_result)));
    if (load_result == LoadResult::Unavailable) {
        result.error = "Router model '" + router_info->id + "' unavailable";
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }
    if (load_result == LoadResult::Deferred) {
        result.error = "Router model '" + router_info->id + "' load deferred";
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Passthrough: no TEXT sub-models → call router directly ──
    // Vision sub-models are not part of the text-routing pool; they are
    // dispatched through PhysicalPerceptionPrimitivesLoop. So a registry
    // that holds only vision sub-models is functionally a passthrough.
    if (registry_.getTextSubModels().empty()) {
        loader_.markInUse(router_info->id);

        auto it = backends_.find(router_info->id);
        if (it == backends_.end()) {
            loader_.markIdle(router_info->id);
            throw std::runtime_error(
                "No registered backend for router '" + router_info->id
                + "' — check createBackendForModel() in bootstrap.cpp.");
        }

        GenerationOptions opts;
        opts.timeout_ms = config_.generate_timeout_ms;

        LOG_DEBUG("MMO_ORCH", "[TRACE] calling backend->generate() for passthrough (timeout=" + std::to_string(opts.timeout_ms) + "ms)");
        GenerationResult gen = it->second->generate(ctx.prompt, opts);
        LOG_DEBUG("MMO_ORCH", "[TRACE] backend->generate() returned (success=" + std::string(gen.success ? "true" : "false") + ")");
        loader_.markIdle(router_info->id);

        if (gen.success) {
            result.success  = true;
            result.response = gen.text;
        } else {
            result.error = "Router passthrough failed: " + gen.error;
            LOG_ERROR("MMO_ORCH", result.error);
        }
        return result;
    }

    // ── Step 3: Build and send route request ──
    loader_.markInUse(router_info->id);

    std::string scope_json = buildRouterScope(ctx);
    RequestEnvelope route_env = buildRouteRequest(
        ctx.request_id, ctx.session_id, ctx.turn_id,
        router_info->id, scope_json, ctx.prompt);

    auto route_violation = validateRequest(route_env);
    if (route_violation) {
        loader_.markIdle(router_info->id);
        throw std::runtime_error(
            "Orchestrator: route envelope validation failed: "
            + route_violation->message);
    }

    ResponseEnvelope route_resp = callBackend(
        *router_info, "/api/mmo/route", route_env, config_.route_timeout_ms);

    // Validate response correlation using Contracts
    auto resp_violation = validateResponse(
        route_resp, ctx.request_id, router_info->id);
    if (resp_violation) {
        loader_.markIdle(router_info->id);
        throw std::runtime_error(
            "Orchestrator: route response validation failed: "
            + resp_violation->message);
    }

    if (route_resp.status != ResponseStatus::Ok) {
        loader_.markIdle(router_info->id);
        result.error = "Router returned " +
            std::string(route_resp.status == ResponseStatus::Refuse
                ? "refusal: " + route_resp.refusal
                : "error: " + route_resp.error);
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Step 4: Parse route decision via ModelRouter ──
    ParsedRouteResult parsed = router_.parseRouteResponse(route_resp);
    if (!parsed.success) {
        loader_.markIdle(router_info->id);
        if (!parsed.refusal.empty()) {
            result.error = "Router refused: " + parsed.refusal;
        } else {
            result.error = "Route parsing failed: " + parsed.error;
        }
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    RouteDecision& decision = parsed.decision;
    result.sub_model_id = decision.sub_model_id;

    LOG_DEBUG("MMO_ORCH", "Route decision: sub_model_id='"
              + decision.sub_model_id + "' confidence="
              + std::to_string(parsed.confidence.overall));

    // ── Step 5: Validate sub-model exists in registry ──
    const ModelInfo* sub_info = registry_.getModelById(decision.sub_model_id);
    if (!sub_info) {
        loader_.markIdle(router_info->id);
        result.error = "Router selected unknown sub_model_id '"
                       + decision.sub_model_id + "'";
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Step 6: Ensure sub-model is loaded ──
    LoadResult sub_load = loader_.ensureLoaded(decision.sub_model_id);
    if (sub_load == LoadResult::Unavailable) {
        loader_.markIdle(router_info->id);
        result.error = "Sub-model '" + decision.sub_model_id + "' unavailable";
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }
    if (sub_load == LoadResult::Deferred) {
        loader_.markIdle(router_info->id);
        result.error = "Sub-model '" + decision.sub_model_id + "' load deferred";
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Step 7: Call sub-model with composed generation only ──
    loader_.markInUse(decision.sub_model_id);

    RequestEnvelope gen_env = buildGenerateRequest(
        ctx.request_id, ctx.session_id, ctx.turn_id,
        sub_info->id, decision.composed_generation);

    auto gen_violation = validateRequest(gen_env);
    if (gen_violation) {
        loader_.markIdle(decision.sub_model_id);
        loader_.markIdle(router_info->id);
        throw std::runtime_error(
            "Orchestrator: generate envelope validation failed: "
            + gen_violation->message);
    }

    ResponseEnvelope sub_resp = callBackend(
        *sub_info, "/api/mmo/generate", gen_env, config_.generate_timeout_ms);

    loader_.markIdle(decision.sub_model_id);

    auto sub_violation = validateResponse(
        sub_resp, ctx.request_id, decision.sub_model_id);
    if (sub_violation) {
        loader_.markIdle(router_info->id);
        throw std::runtime_error(
            "Orchestrator: sub-model response validation failed: "
            + sub_violation->message);
    }

    if (sub_resp.status != ResponseStatus::Ok) {
        loader_.markIdle(router_info->id);
        result.error = "Sub-model '" + decision.sub_model_id + "' returned " +
            std::string(sub_resp.status == ResponseStatus::Refuse
                ? "refusal: " + sub_resp.refusal
                : "error: " + sub_resp.error);
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Step 8: Feed sub-model output back to router (synthesize) ──
    RequestEnvelope synth_env = buildSynthesizeRequest(
        ctx.request_id, ctx.session_id, ctx.turn_id,
        router_info->id, sub_resp.result);

    auto synth_violation = validateRequest(synth_env);
    if (synth_violation) {
        loader_.markIdle(router_info->id);
        throw std::runtime_error(
            "Orchestrator: synthesize envelope validation failed: "
            + synth_violation->message);
    }

    ResponseEnvelope synth_resp = callBackend(
        *router_info, "/api/mmo/synthesize", synth_env, config_.synthesize_timeout_ms);

    loader_.markIdle(router_info->id);

    auto synth_resp_violation = validateResponse(
        synth_resp, ctx.request_id, router_info->id);
    if (synth_resp_violation) {
        throw std::runtime_error(
            "Orchestrator: synthesize response validation failed: "
            + synth_resp_violation->message);
    }

    if (synth_resp.status != ResponseStatus::Ok) {
        result.error = "Synthesize step returned " +
            std::string(synth_resp.status == ResponseStatus::Refuse
                ? "refusal: " + synth_resp.refusal
                : "error: " + synth_resp.error);
        LOG_ERROR("MMO_ORCH", result.error);
        return result;
    }

    // ── Step 9: Return final synthesized response ──
    result.success  = true;
    result.response = synth_resp.result;

    LOG_DEBUG("MMO_ORCH", "Request '" + ctx.request_id
              + "' completed via sub-model '" + decision.sub_model_id + "'");

    return result;
}

// =========================================================
// shutdown
// =========================================================

void Orchestrator::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Flush any pending correction tuples before unloading models
    size_t flushed = CorrectionTupleCollector::instance().flush(
        config_.correction_output_path);
    if (flushed > 0) {
        LOG_DEBUG("MMO_ORCH", "Flushed " + std::to_string(flushed)
                  + " correction tuples to disk");
    }

    // Flush training examples (positive + negative signal)
    size_t trainFlushed = ToolTrainingParser::instance().flush(
        "tool_training_examples.jsonl");
    if (trainFlushed > 0) {
        LOG_DEBUG("MMO_ORCH", "Flushed " + std::to_string(trainFlushed)
                  + " training examples to disk");
    }

    loader_.unloadAll();
    LOG_DEBUG("MMO_ORCH", "Orchestrator shutdown complete");
}

// =========================================================
// buildRouterScope — memory enrichment for route request
// =========================================================

std::string Orchestrator::buildRouterScope(const RequestContext& ctx) const {
    // Drain preprocessing buffer into working context before building scope
    GRIM::MemoryBufferRotation::instance().mergeToWorking();

    if (!memory_) {
        return ctx.metadata_json;
    }

    auto retrieval = memory_->retrieveForPrompt(ctx.prompt);

    nlohmann::json scope;
    // Preserve any caller-provided metadata
    if (!ctx.metadata_json.empty()) {
        auto parsed = nlohmann::json::parse(
            ctx.metadata_json, nullptr, false);
        if (!parsed.is_discarded()) {
            scope = std::move(parsed);
        }
    }

    // Add context snapshot
    nlohmann::json ctx_json;
    ctx_json["current_mood"]          = retrieval.context.currentMood;
    ctx_json["conversation_depth"]    = retrieval.context.conversationDepth;
    ctx_json["last_nlp_category"]     = retrieval.context.lastNlpCategory;
    ctx_json["consecutive_commands"]  = retrieval.context.consecutiveCommands;
    ctx_json["recent_intents"]        = retrieval.context.recentIntents;
    ctx_json["recent_commands"]       = retrieval.context.recentCommands;
    scope["context"] = std::move(ctx_json);

    // Add ranked memories
    nlohmann::json mem_array = nlohmann::json::array();
    for (const auto& m : retrieval.memories) {
        nlohmann::json entry;
        entry["id"]         = m.id;
        entry["raw"]        = m.raw;
        entry["normalized"] = m.normalized;
        entry["confidence"] = m.confidence;
        entry["tags"]       = m.tags;
        mem_array.push_back(std::move(entry));
    }
    scope["memories"] = std::move(mem_array);

    // Add breadcrumbs for diagnostics
    nlohmann::json bc_array = nlohmann::json::array();
    for (const auto& bc : retrieval.breadcrumbs) {
        nlohmann::json b;
        b["query"]         = bc.query;
        b["results_found"] = bc.results_found;
        bc_array.push_back(std::move(b));
    }
    scope["retrieval_breadcrumbs"] = std::move(bc_array);

    return scope.dump();
}

// =========================================================
// callBackend — dispatch to model backend via IGenerationBackend
//
// All models MUST have a registered backend. No fallback.
// =========================================================

ResponseEnvelope Orchestrator::callBackend(
    const ModelInfo& model,
    const std::string& endpoint,
    const RequestEnvelope& envelope,
    int timeout_ms) {

    auto it = backends_.find(model.id);
    if (it == backends_.end()) {
        throw std::runtime_error(
            "No registered backend for model '" + model.id
            + "' — all models must be registered at bootstrap. "
            "Check createBackendForModel() in bootstrap.cpp.");
    }

    IGenerationBackend* backend = it->second.get();

    GenerationOptions opts;
    opts.timeout_ms = timeout_ms;
    if (!envelope.scope.empty())    opts.metadata_json = envelope.scope;

    // Pass full envelope so MMO-aware backends can use
    // the structured endpoint instead of /api/chat.
    nlohmann::json env_body;
    env_body["schema_version"]  = envelope.schema_version;
    env_body["request_id"]      = envelope.request_id;
    env_body["session_id"]      = envelope.session_id;
    env_body["turn_id"]         = envelope.turn_id;
    env_body["target_model_id"] = envelope.target_model_id;
    env_body["task"]            = envelope.task;
    env_body["payload"]         = envelope.payload;
    if (!envelope.scope.empty())         env_body["scope"]         = envelope.scope;
    if (!envelope.constraints.empty())   env_body["constraints"]   = envelope.constraints;
    if (!envelope.output_schema.empty()) env_body["output_schema"] = envelope.output_schema;
    if (envelope.max_length > 0)         env_body["max_length"]    = envelope.max_length;
    opts.envelope_json = env_body.dump();
    opts.mmo_endpoint  = endpoint;

    GenerationResult gen_result = backend->generate(
        envelope.payload, opts);

    return resultToEnvelope(gen_result, envelope);
}

// =========================================================
// resultToEnvelope — convert GenerationResult → ResponseEnvelope
// =========================================================

ResponseEnvelope Orchestrator::resultToEnvelope(
    const GenerationResult& gen_result,
    const RequestEnvelope& envelope) const {

    ResponseEnvelope response;
    response.schema_version  = envelope.schema_version;
    response.request_id      = envelope.request_id;
    response.target_model_id = envelope.target_model_id;

    if (gen_result.success) {
        // Try to parse as MMO envelope JSON first
        auto j = nlohmann::json::parse(gen_result.text, nullptr, false);
        if (!j.is_discarded() && j.contains("status")) {
            // Backend returned structured MMO envelope
            response.schema_version  = j.value("schema_version", 1u);
            if (j.contains("request_id"))
                response.request_id  = j.value("request_id", std::string{});
            if (j.contains("target_model_id"))
                response.target_model_id = j.value("target_model_id", std::string{});

            std::string status_str = j.value("status", std::string{"error"});
            if (status_str == "ok") {
                response.status = ResponseStatus::Ok;
                response.result = j.value("result", std::string{});
            } else if (status_str == "refuse") {
                response.status  = ResponseStatus::Refuse;
                response.refusal = j.value("refusal", std::string{});
            } else {
                response.status = ResponseStatus::Error;
                response.error  = j.value("error", std::string{"unknown error"});
            }
        } else {
            // Backend returned raw text — wrap as Ok result
            response.status = ResponseStatus::Ok;
            response.result = gen_result.text;
        }
    } else {
        response.status = ResponseStatus::Error;
        response.error  = gen_result.error;
    }

    return response;
}

// =========================================================
// evaluateToolGap — check if a capability is missing
// =========================================================

std::optional<ToolGapProposal> Orchestrator::evaluateToolGap(
    const std::string& capability,
    const std::string& request_id) const {
    return tool_gap_planner_.evaluate(capability, request_id);
}

} // namespace GRIM::MMO
