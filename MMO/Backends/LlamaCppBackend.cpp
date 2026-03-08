// LlamaCppBackend — IGenerationBackend for llama.cpp server
// See LlamaCppBackend.hpp for interface documentation.
//
// llama.cpp server exposes an OpenAI-compatible API:
//   POST /v1/chat/completions  — chat completions
//   GET  /health               — health check
//======================================================//
#include "LlamaCppBackend.hpp"

#include "../../logger.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

LlamaCppBackend::LlamaCppBackend(const std::string& url,
                                 const std::string& model_id,
                                 const std::string& model_name)
    : url_(url)
    , model_id_(model_id)
    , model_name_(model_name) {
    if (url_.empty()) {
        throw std::runtime_error(
            "LlamaCppBackend: url is empty — caller MUST provide valid url");
    }
    if (model_id_.empty()) {
        throw std::runtime_error(
            "LlamaCppBackend: model_id is empty — caller MUST provide valid model_id");
    }
    if (model_name_.empty()) {
        throw std::runtime_error(
            "LlamaCppBackend: model_name is empty — caller MUST provide valid model name");
    }
}

// =========================================================
// generate — single-turn via /v1/chat/completions
// =========================================================

GenerationResult LlamaCppBackend::generate(
    const std::string& prompt,
    const GenerationOptions& options) {

    return generateWithHistory(prompt, {}, options);
}

// =========================================================
// generateWithHistory — multi-turn via /v1/chat/completions
//
// Uses the OpenAI-compatible chat completions format that
// llama.cpp server supports.
// =========================================================

GenerationResult LlamaCppBackend::generateWithHistory(
    const std::string& prompt,
    const std::vector<HistoryEntry>& history,
    const GenerationOptions& options) {

    GenerationResult result;

    // Build messages array (OpenAI format)
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

    // Build request body (OpenAI-compatible format)
    nlohmann::json body;
    body["model"]    = model_name_;
    body["messages"] = messages;
    body["stream"]   = false;

    if (options.max_tokens > 0)     body["max_tokens"]   = options.max_tokens;
    if (options.temperature > 0.0f) body["temperature"]   = options.temperature;
    if (options.top_p > 0.0f)       body["top_p"]         = options.top_p;

    int timeout = options.timeout_ms > 0 ? options.timeout_ms : 60000;
    std::string endpoint = url_ + "/v1/chat/completions";

    LOG_DEBUG("MMO_LLAMACPP", "POST " + endpoint + " model=" + model_name_);

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
        LOG_ERROR("MMO_LLAMACPP", result.error);
        return result;
    }

    auto j = nlohmann::json::parse(http_resp.text, nullptr, false);
    if (j.is_discarded()) {
        result.error = "Invalid JSON from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_LLAMACPP", result.error);
        return result;
    }

    // OpenAI chat completions response:
    // { "choices": [{ "message": { "role": "assistant", "content": "..." } }], ... }
    if (j.contains("choices") && j["choices"].is_array()
        && !j["choices"].empty()
        && j["choices"][0].contains("message")
        && j["choices"][0]["message"].contains("content")) {
        result.success = true;
        result.text    = j["choices"][0]["message"]["content"].get<std::string>();
    } else {
        result.error = "Unexpected response format from " + endpoint;
        LOG_ERROR("MMO_LLAMACPP", result.error);
        return result;
    }

    // Extract token usage if reported
    if (j.contains("usage") && j["usage"].contains("completion_tokens")) {
        result.tokens_used = j["usage"]["completion_tokens"].get<int>();
    }

    return result;
}

// =========================================================
// isAvailable — health check via /health
// =========================================================

bool LlamaCppBackend::isAvailable() const {
    try {
        cpr::Response resp = cpr::Get(
            cpr::Url{url_ + "/health"},
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

std::string LlamaCppBackend::getBackendId() const {
    return "llamacpp:" + model_id_;
}

BackendType LlamaCppBackend::getBackendType() const {
    return BackendType::LlamaCpp;
}

} // namespace GRIM::MMO
