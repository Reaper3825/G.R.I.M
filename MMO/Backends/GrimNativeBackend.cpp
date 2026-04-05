// GrimNativeBackend — IGenerationBackend for grim_text_server.exe
// See GrimNativeBackend.hpp for interface documentation.
//======================================================//
#include "GrimNativeBackend.hpp"

#include "../../logger.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

GrimNativeBackend::GrimNativeBackend(const std::string& url,
                                     const std::string& model_id)
    : url_(url)
    , model_id_(model_id) {
    if (url_.empty()) {
        throw std::runtime_error(
            "GrimNativeBackend: url is empty — caller MUST provide valid url");
    }
    if (model_id_.empty()) {
        throw std::runtime_error(
            "GrimNativeBackend: model_id is empty — caller MUST provide valid model_id");
    }
}

// =========================================================
// generate — single-turn generation via /api/chat
// =========================================================

GenerationResult GrimNativeBackend::generate(
    const std::string& prompt,
    const GenerationOptions& options) {

    // ── MMO envelope path ──
    // When the Orchestrator populates envelope_json + mmo_endpoint,
    // POST the full RequestEnvelope to the MMO endpoint and parse
    // the ResponseEnvelope JSON back.
    if (!options.envelope_json.empty() && !options.mmo_endpoint.empty()) {
        return generateEnvelope(options);
    }

    return generateWithHistory(prompt, {}, options);
}

// =========================================================
// generateWithHistory — multi-turn generation via /api/chat
//
// Uses the Ollama-compatible /api/chat format that grim-text
// server already supports.
// =========================================================

GenerationResult GrimNativeBackend::generateWithHistory(
    const std::string& prompt,
    const std::vector<HistoryEntry>& history,
    const GenerationOptions& options) {

    GenerationResult result;

    // Build messages array
    nlohmann::json messages = nlohmann::json::array();

    for (const auto& entry : history) {
        messages.push_back({
            {"role",    entry.role},
            {"content", entry.content}
        });
    }

    // Add current prompt as user message
    messages.push_back({
        {"role",    "user"},
        {"content", prompt}
    });

    // Build request body (Ollama-compatible format)
    nlohmann::json body;
    body["model"]    = "grim-text";
    body["messages"] = messages;
    body["stream"]   = false;

    if (options.max_tokens > 0)    body["max_tokens"]   = options.max_tokens;
    if (options.temperature > 0.0f) body["temperature"]  = options.temperature;
    if (options.top_p > 0.0f)      body["top_p"]        = options.top_p;
    if (options.top_k > 0)         body["top_k"]        = options.top_k;

    int timeout = options.timeout_ms > 0 ? options.timeout_ms : 30000;
    std::string endpoint = url_ + "/api/chat";

    LOG_DEBUG("MMO_GRIM_NATIVE", "[TRACE] POST " + endpoint + " (timeout=" + std::to_string(timeout) + "ms)");

    cpr::Response http_resp = cpr::Post(
        cpr::Url{endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()},
        cpr::Timeout{timeout}
    );
    LOG_DEBUG("MMO_GRIM_NATIVE", "[TRACE] POST returned status=" + std::to_string(http_resp.status_code)
              + " elapsed=" + std::to_string(http_resp.elapsed) + "s");

    if (http_resp.status_code != 200) {
        result.error = "HTTP " + std::to_string(http_resp.status_code)
                       + " from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_GRIM_NATIVE", result.error);
        return result;
    }

    auto j = nlohmann::json::parse(http_resp.text, nullptr, false);
    if (j.is_discarded()) {
        result.error = "Invalid JSON from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_GRIM_NATIVE", result.error);
        return result;
    }

    // Parse Ollama-compatible response format (canonical: /api/chat)
    if (j.contains("message") && j["message"].contains("content")) {
        result.success = true;
        result.text    = j["message"]["content"].get<std::string>();
    } else {
        result.error = "Unexpected response format from " + endpoint
                       + " — expected {\"message\":{\"content\":\"...\"}}";
        LOG_ERROR("MMO_GRIM_NATIVE", result.error);
        return result;
    }

    if (j.contains("eval_count")) {
        result.tokens_used = j["eval_count"].get<int>();
    }

    return result;
}

// =========================================================
// generateEnvelope — MMO envelope-aware generation
//
// Posts the full RequestEnvelope JSON to the appropriate
// MMO endpoint (/api/mmo/route, /api/mmo/generate, etc.)
// and parses the ResponseEnvelope JSON response.
// =========================================================

GenerationResult GrimNativeBackend::generateEnvelope(
    const GenerationOptions& options) {

    GenerationResult result;

    int timeout = options.timeout_ms > 0 ? options.timeout_ms : 30000;
    std::string endpoint = url_ + options.mmo_endpoint;

    LOG_DEBUG("MMO_GRIM_NATIVE", "POST " + endpoint + " (envelope)");

    cpr::Response http_resp = cpr::Post(
        cpr::Url{endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{options.envelope_json},
        cpr::Timeout{timeout}
    );

    if (http_resp.status_code != 200) {
        result.error = "HTTP " + std::to_string(http_resp.status_code)
                       + " from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_GRIM_NATIVE", result.error);
        return result;
    }

    auto j = nlohmann::json::parse(http_resp.text, nullptr, false);
    if (j.is_discarded()) {
        result.error = "Invalid JSON from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_GRIM_NATIVE", result.error);
        return result;
    }

    // The response is already a ResponseEnvelope JSON — return it
    // as-is so resultToEnvelope() in the Orchestrator can parse it.
    result.success = true;
    result.text    = http_resp.text;

    return result;
}

// =========================================================
// isAvailable — health check
// =========================================================

bool GrimNativeBackend::isAvailable() const {
    try {
        cpr::Response resp = cpr::Get(
            cpr::Url{url_ + "/api/tags"},
            cpr::Timeout{2000}
        );
        return resp.status_code == 200;
    } catch (...) {
        return false;
    }
}

// =========================================================
// getBackendId / getBackendType
// =========================================================

std::string GrimNativeBackend::getBackendId() const {
    return "grim_text:" + model_id_;
}

BackendType GrimNativeBackend::getBackendType() const {
    return BackendType::GrimTextServer;
}

} // namespace GRIM::MMO
