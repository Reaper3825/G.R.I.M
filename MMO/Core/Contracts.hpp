// MMO Contracts — route/synthesize/sub-model validation
//
// Extends the raw transport envelope types in MMD.hpp with
// schema validation, structured error codes, and helper
// builders for route/generate/synthesize envelopes.
//
// All envelope construction and validation goes through
// these functions so that invalid contracts fail closed.
//======================================================//
#pragma once

#include "../Shared/MMD.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Structured MMO error codes
// =========================================================
enum class ContractError : uint8_t {
    None                    = 0,
    MissingRequestId        = 1,
    MissingSessionId        = 2,
    MissingTargetModel      = 3,
    MissingTask             = 4,
    MissingPayload          = 5,
    SchemaVersionMismatch   = 6,
    RequestIdMismatch       = 7,
    TargetModelMismatch     = 8,
    InvalidResponseStatus   = 9,
    EmptyResult             = 10,
    MalformedJson           = 11,
    TimeoutExceeded         = 12,
    UnknownSubModel         = 13,
    RouterRefused           = 14,
    SubModelRefused         = 15,
    SynthesisRefused        = 16,
};

// =========================================================
// ContractViolation — detailed validation failure
// =========================================================
struct ContractViolation {
    ContractError code    = ContractError::None;
    std::string   field;    // which field failed
    std::string   message;  // human-readable explanation
};

// =========================================================
// Validation — checks a request or response envelope
// Returns nullopt on success, or a violation on failure.
// =========================================================

// Validate a RequestEnvelope before sending to any model.
std::optional<ContractViolation> validateRequest(const RequestEnvelope& env);

// Validate a ResponseEnvelope received from a model.
// expected_request_id / expected_target_model_id are from the
// in-flight RequestEnvelope — responses must correlate.
std::optional<ContractViolation> validateResponse(
    const ResponseEnvelope& env,
    const std::string& expected_request_id,
    const std::string& expected_target_model_id);

// =========================================================
// Envelope builders — construct validated envelopes
//
// These throw on programming errors (missing required fields)
// per Rule 20 — fail loud, no silent fallbacks.
// =========================================================

// Build a route request envelope.
// task = "route", payload = prompt/context, scope = metadata_json
RequestEnvelope buildRouteRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& router_model_id,
    const std::string& metadata_json,
    const std::string& prompt);

// Build a sub-model generation request envelope.
// task = "generate", payload = composed_generation only
RequestEnvelope buildGenerateRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& sub_model_id,
    const std::string& composed_generation,
    int max_length = 0);

// Build a synthesis request envelope.
// task = "synthesize", payload = sub-model results JSON
RequestEnvelope buildSynthesizeRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& router_model_id,
    const std::string& sub_model_results_json);

// =========================================================
// Composed generation template builder
//
// Builds the strict structured prompt sent to sub-models:
// TASK, SCOPE, ALLOWED_ASSUMPTIONS, OUTPUT_SCHEMA,
// REFUSE_IF, STYLE, MAX_LENGTH
// =========================================================
// =========================================================
// RouteDecision — parsed from router's route response
//
// Moved here from Orchestrator.hpp so ModelRouter and
// Orchestrator can both include Contracts.hpp without
// circular dependencies.
// =========================================================
struct RouteDecision {
    std::string sub_model_id;
    std::string composed_generation;
    std::string diagnostics;        // optional JSON string
};

struct ComposedGenerationSpec {
    std::string task;
    std::string scope;
    std::string allowed_assumptions;
    std::string output_schema;
    std::string refuse_if;
    std::string style;
    int         max_length = 0;
    std::string injected_context;   // router-injected context block
};

// Render a ComposedGenerationSpec into the tagged text format
// understood by sub-models.
std::string renderComposedGeneration(const ComposedGenerationSpec& spec);

// Parse a composed generation string back into spec (for diagnostics).
std::optional<ComposedGenerationSpec> parseComposedGeneration(const std::string& text);

} // namespace GRIM::MMO
