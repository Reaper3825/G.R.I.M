// ExternalBackend — IGenerationBackend for arbitrary HTTP endpoints
// See ExternalBackend.hpp for interface documentation.
//
// Sends a JSON POST with OpenAI-compatible message format.
// Parses response as OpenAI-compatible JSON first; falls back
// to raw text if the response is not structured JSON.
//======================================================//
#include "ExternalBackend.hpp"

#include "../../logger.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

ExternalBackend::ExternalBackend(const std::string& url,
                                 const std::string& model_id,
                                 const std::string& endpoint_path)
    : url_(url)
    , model_id_(model_id)
    , endpoint_path_(endpoint_path) {
    if (url_.empty()) {
        throw std::runtime_error(
            "ExternalBackend: url is empty — caller MUST provide valid url");
    }
    if (model_id_.empty()) {
        throw std::runtime_error(
            "ExternalBackend: model_id is empty — caller MUST provide valid model_id");
    }
    if (endpoint_path_.empty()) {
        throw std::runtime_error(
            "ExternalBackend: endpoint_path is empty — caller MUST provide valid endpoint path");
    }
}

// =========================================================
// generate — single-turn
// =========================================================

GenerationResult ExternalBackend::generate(
    const std::string& prompt,
    const GenerationOptions& options) {

    return generateWithHistory(prompt, {}, options);
}

// =========================================================
// generateWithHistory — multi-turn via configured endpoint
//
// Sends OpenAI-compatible message format. Parses response
// as OpenAI JSON first, then Ollama format, then raw text.
// =========================================================

GenerationResult ExternalBackend::generateWithHistory(
    const std::string& prompt,
    const std::vector<HistoryEntry>& history,
    const GenerationOptions& options) {

    GenerationResult result;

    // Build messages array (OpenAI format — widely supported)
    nlohmann::json messages = nlohmann::json::array();

    for (const auto& entry : history) {
        messages.push_back({
            {"role",    entry.role},
            {"content", entry.content}
        });
    }

    messages.push_back({
        {"role",    "user"},
        {"content", prompt}
    });

    nlohmann::json body;
    body["messages"] = messages;
    body["stream"]   = false;

    if (options.max_tokens > 0)     body["max_tokens"]   = options.max_tokens;
    if (options.temperature > 0.0f) body["temperature"]   = options.temperature;
    if (options.top_p > 0.0f)       body["top_p"]         = options.top_p;
    if (options.top_k > 0)          body["top_k"]         = options.top_k;

    int timeout = options.timeout_ms > 0 ? options.timeout_ms : 60000;
    std::string endpoint = url_ + endpoint_path_;

    LOG_DEBUG("MMO_EXTERNAL", "POST " + endpoint);

    cpr::Response http_resp = cpr::Post(
        cpr::Url{endpoint},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{body.dump()},
        cpr::Timeout{timeout}
    );

    if (http_resp.status_code != 200) {
        result.error = "HTTP " + std::to_string(http_resp.status_code)
                       + " from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_EXTERNAL", result.error);
        return result;
    }

    // Attempt structured JSON parse
    auto j = nlohmann::json::parse(http_resp.text, nullptr, false);

    if (!j.is_discarded()) {
        // Try OpenAI format: { "choices": [{ "message": { "content": "..." } }] }
        if (j.contains("choices") && j["choices"].is_array()
            && !j["choices"].empty()
            && j["choices"][0].contains("message")
            && j["choices"][0]["message"].contains("content")) {
            result.success = true;
            result.text    = j["choices"][0]["message"]["content"].get<std::string>();

            if (j.contains("usage") && j["usage"].contains("completion_tokens")) {
                result.tokens_used = j["usage"]["completion_tokens"].get<int>();
            }
            return result;
        }

        // Try Ollama format: { "message": { "content": "..." } }
        if (j.contains("message") && j["message"].contains("content")) {
            result.success = true;
            result.text    = j["message"]["content"].get<std::string>();

            if (j.contains("eval_count")) {
                result.tokens_used = j["eval_count"].get<int>();
            }
            return result;
        }

        // Try simple text field: { "text": "..." } or { "response": "..." }
        if (j.contains("text") && j["text"].is_string()) {
            result.success = true;
            result.text    = j["text"].get<std::string>();
            return result;
        }
        if (j.contains("response") && j["response"].is_string()) {
            result.success = true;
            result.text    = j["response"].get<std::string>();
            return result;
        }
    }

    // Raw text fallback — treat entire response body as generated text
    if (!http_resp.text.empty()) {
        result.success = true;
        result.text    = http_resp.text;
        return result;
    }

    result.error = "Empty response from " + endpoint;
    LOG_ERROR("MMO_EXTERNAL", result.error);
    return result;
}

// =========================================================
// isAvailable — attempt a GET to the base URL
// =========================================================

bool ExternalBackend::isAvailable() const {
    try {
        cpr::Response resp = cpr::Get(
            cpr::Url{url_ + "/"},
            cpr::Timeout{2000}
        );
        // Accept any 2xx status as "available"
        return resp.status_code >= 200 && resp.status_code < 300;
    } catch (...) {
        return false;
    }
}

// =========================================================
// getBackendId / getBackendType
// =========================================================

std::string ExternalBackend::getBackendId() const {
    return "external:" + model_id_;
}

BackendType ExternalBackend::getBackendType() const {
    return BackendType::External;
}

} // namespace GRIM::MMO
