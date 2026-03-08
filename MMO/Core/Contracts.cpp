// MMO Contracts implementation — validation + builders
//======================================================//

#include "Contracts.hpp"
#include <stdexcept>
#include <sstream>

namespace GRIM::MMO {

// ─── Validation ───────────────────────────────────────────

std::optional<ContractViolation> validateRequest(const RequestEnvelope& env) {
    if (env.request_id.empty())
        return ContractViolation{ContractError::MissingRequestId, "request_id", "request_id is empty"};
    if (env.session_id.empty())
        return ContractViolation{ContractError::MissingSessionId, "session_id", "session_id is empty"};
    if (env.target_model_id.empty())
        return ContractViolation{ContractError::MissingTargetModel, "target_model_id", "target_model_id is empty"};
    if (env.task.empty())
        return ContractViolation{ContractError::MissingTask, "task", "task is empty"};
    if (env.payload.empty())
        return ContractViolation{ContractError::MissingPayload, "payload", "payload is empty"};
    return std::nullopt;
}

std::optional<ContractViolation> validateResponse(
    const ResponseEnvelope& env,
    const std::string& expected_request_id,
    const std::string& expected_target_model_id) {

    if (env.schema_version != 1)
        return ContractViolation{ContractError::SchemaVersionMismatch, "schema_version",
            "expected schema_version=1, got " + std::to_string(env.schema_version)};
    if (env.request_id != expected_request_id)
        return ContractViolation{ContractError::RequestIdMismatch, "request_id",
            "expected '" + expected_request_id + "', got '" + env.request_id + "'"};
    if (env.target_model_id != expected_target_model_id)
        return ContractViolation{ContractError::TargetModelMismatch, "target_model_id",
            "expected '" + expected_target_model_id + "', got '" + env.target_model_id + "'"};
    if (env.status == ResponseStatus::Ok && env.result.empty())
        return ContractViolation{ContractError::EmptyResult, "result",
            "status is Ok but result is empty"};
    return std::nullopt;
}

// ─── Envelope builders ───────────────────────────────────

RequestEnvelope buildRouteRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& router_model_id,
    const std::string& metadata_json,
    const std::string& prompt) {

    if (request_id.empty()) throw std::runtime_error("buildRouteRequest: request_id is empty");
    if (router_model_id.empty()) throw std::runtime_error("buildRouteRequest: router_model_id is empty");

    RequestEnvelope env;
    env.schema_version  = 1;
    env.request_id      = request_id;
    env.session_id      = session_id;
    env.turn_id         = turn_id;
    env.target_model_id = router_model_id;
    env.task            = "route";
    env.scope           = metadata_json;
    env.payload         = prompt;
    return env;
}

RequestEnvelope buildGenerateRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& sub_model_id,
    const std::string& composed_generation,
    int max_length) {

    if (request_id.empty()) throw std::runtime_error("buildGenerateRequest: request_id is empty");
    if (sub_model_id.empty()) throw std::runtime_error("buildGenerateRequest: sub_model_id is empty");
    if (composed_generation.empty()) throw std::runtime_error("buildGenerateRequest: composed_generation is empty");

    RequestEnvelope env;
    env.schema_version  = 1;
    env.request_id      = request_id;
    env.session_id      = session_id;
    env.turn_id         = turn_id;
    env.target_model_id = sub_model_id;
    env.task            = "generate";
    env.payload         = composed_generation;
    env.max_length      = max_length;
    return env;
}

RequestEnvelope buildSynthesizeRequest(
    const std::string& request_id,
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& router_model_id,
    const std::string& sub_model_results_json) {

    if (request_id.empty()) throw std::runtime_error("buildSynthesizeRequest: request_id is empty");
    if (router_model_id.empty()) throw std::runtime_error("buildSynthesizeRequest: router_model_id is empty");

    RequestEnvelope env;
    env.schema_version  = 1;
    env.request_id      = request_id;
    env.session_id      = session_id;
    env.turn_id         = turn_id;
    env.target_model_id = router_model_id;
    env.task            = "synthesize";
    env.payload         = sub_model_results_json;
    return env;
}

// ─── ComposedGeneration ──────────────────────────────────

std::string renderComposedGeneration(const ComposedGenerationSpec& spec) {
    if (spec.task.empty()) throw std::runtime_error("renderComposedGeneration: task is empty");

    std::ostringstream out;
    out << "TASK: " << spec.task << "\n";
    if (!spec.scope.empty())
        out << "SCOPE: " << spec.scope << "\n";
    if (!spec.allowed_assumptions.empty())
        out << "ALLOWED_ASSUMPTIONS: " << spec.allowed_assumptions << "\n";
    if (!spec.output_schema.empty())
        out << "OUTPUT_SCHEMA: " << spec.output_schema << "\n";
    if (!spec.refuse_if.empty())
        out << "REFUSE_IF: " << spec.refuse_if << "\n";
    if (!spec.style.empty())
        out << "STYLE: " << spec.style << "\n";
    if (spec.max_length > 0)
        out << "MAX_LENGTH: " << spec.max_length << "\n";
    if (!spec.injected_context.empty())
        out << "\n---\n" << spec.injected_context;
    return out.str();
}

static std::string extractField(const std::string& text, const std::string& label) {
    std::string prefix = label + ": ";
    auto pos = text.find(prefix);
    if (pos == std::string::npos) return "";
    auto start = pos + prefix.size();
    auto end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    return text.substr(start, end - start);
}

std::optional<ComposedGenerationSpec> parseComposedGeneration(const std::string& text) {
    ComposedGenerationSpec spec;
    spec.task = extractField(text, "TASK");
    if (spec.task.empty()) return std::nullopt;
    spec.scope                = extractField(text, "SCOPE");
    spec.allowed_assumptions  = extractField(text, "ALLOWED_ASSUMPTIONS");
    spec.output_schema        = extractField(text, "OUTPUT_SCHEMA");
    spec.refuse_if            = extractField(text, "REFUSE_IF");
    spec.style                = extractField(text, "STYLE");
    std::string ml            = extractField(text, "MAX_LENGTH");
    if (!ml.empty()) {
        try { spec.max_length = std::stoi(ml); } catch (...) {}
    }
    auto sep = text.find("\n---\n");
    if (sep != std::string::npos) {
        spec.injected_context = text.substr(sep + 5);
    }
    return spec;
}

} // namespace GRIM::MMO
