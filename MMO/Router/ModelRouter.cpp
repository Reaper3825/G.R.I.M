// MMO ModelRouter implementation — JSON parsing of router responses
//======================================================//
#include "ModelRouter.hpp"

#include "../../logger.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// parseRouteResponse — from ResponseEnvelope
// =========================================================

ParsedRouteResult ModelRouter::parseRouteResponse(const ResponseEnvelope& response) const {
    ParsedRouteResult result;

    if (response.status == ResponseStatus::Error) {
        result.error = "Router returned error: " + response.error;
        return result;
    }

    if (response.status == ResponseStatus::Refuse) {
        result.refusal = response.refusal;
        return result;
    }

    // Status == Ok — parse the result JSON
    if (response.result.empty()) {
        result.error = "Router returned Ok but result is empty";
        return result;
    }

    return parseRouteJson(response.result);
}

// =========================================================
// parseRouteJson — from raw JSON text
// =========================================================

ParsedRouteResult ModelRouter::parseRouteJson(const std::string& json_text) const {
    ParsedRouteResult result;

    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::parse_error& e) {
        result.error = std::string("Router response is not valid JSON: ") + e.what();
        return result;
    }

    // ── Required: sub_model_id ──
    if (!doc.contains("sub_model_id") || !doc["sub_model_id"].is_string()) {
        result.error = "Router response missing 'sub_model_id' string field";
        return result;
    }

    std::string sub_model_id = doc["sub_model_id"].get<std::string>();

    // Empty sub_model_id with refusal present = router refused
    if (sub_model_id.empty()) {
        if (doc.contains("refusal") && doc["refusal"].is_string()) {
            result.refusal = doc["refusal"].get<std::string>();
        } else {
            result.refusal = "Router returned empty sub_model_id with no refusal reason";
        }
        return result;
    }

    // ── Required: composed_generation ──
    if (!doc.contains("composed_generation") || !doc["composed_generation"].is_string()) {
        result.error = "Router response missing 'composed_generation' string field";
        return result;
    }

    std::string composed_gen = doc["composed_generation"].get<std::string>();
    if (composed_gen.empty()) {
        result.error = "Router returned empty composed_generation";
        return result;
    }

    // ── Success — populate decision ──
    result.success = true;
    result.decision.sub_model_id = std::move(sub_model_id);
    result.decision.composed_generation = std::move(composed_gen);

    // ── Optional: diagnostics ──
    if (doc.contains("diagnostics")) {
        if (doc["diagnostics"].is_string()) {
            result.decision.diagnostics = doc["diagnostics"].get<std::string>();
        } else {
            result.decision.diagnostics = doc["diagnostics"].dump();
        }
    }

    // ── Optional: confidence scores ──
    if (doc.contains("confidence") && doc["confidence"].is_number()) {
        result.confidence.overall = doc["confidence"].get<float>();
    }
    if (doc.contains("confidence_user_intent") && doc["confidence_user_intent"].is_number()) {
        result.confidence.user_intent = doc["confidence_user_intent"].get<float>();
    }
    if (doc.contains("confidence_domain_match") && doc["confidence_domain_match"].is_number()) {
        result.confidence.domain_match = doc["confidence_domain_match"].get<float>();
    }

    return result;
}

} // namespace GRIM::MMO
