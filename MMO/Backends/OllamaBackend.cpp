// OllamaBackend — IGenerationBackend for Ollama API
// See OllamaBackend.hpp for interface documentation.
//======================================================//
#include "OllamaBackend.hpp"

#include "../../logger.hpp"

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

OllamaBackend::OllamaBackend(const std::string& url,
                               const std::string& model_id,
                               const std::string& ollama_model)
    : url_(url)
    , model_id_(model_id)
    , ollama_model_(ollama_model) {
    if (url_.empty()) {
        throw std::runtime_error(
            "OllamaBackend: url is empty — caller MUST provide valid url");
    }
    if (model_id_.empty()) {
        throw std::runtime_error(
            "OllamaBackend: model_id is empty — caller MUST provide valid model_id");
    }
    if (ollama_model_.empty()) {
        throw std::runtime_error(
            "OllamaBackend: ollama_model is empty — caller MUST provide valid model name");
    }
}

// =========================================================
// generate — single-turn via /api/chat
// =========================================================

GenerationResult OllamaBackend::generate(
    const std::string& prompt,
    const GenerationOptions& options) {

    return generateWithHistory(prompt, {}, options);
}

// =========================================================
// generateWithHistory — multi-turn via /api/chat
// =========================================================

GenerationResult OllamaBackend::generateWithHistory(
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

    messages.push_back({
        {"role",    "user"},
        {"content", prompt}
    });

    // Build Ollama options
    nlohmann::json ollama_options;
    if (options.max_tokens > 0)     ollama_options["num_predict"]  = options.max_tokens;
    if (options.temperature > 0.0f) ollama_options["temperature"]  = options.temperature;
    if (options.top_p > 0.0f)       ollama_options["top_p"]        = options.top_p;
    if (options.top_k > 0)          ollama_options["top_k"]        = options.top_k;
    if (options.seed >= 0)           ollama_options["seed"]         = options.seed;

    nlohmann::json body;
    body["model"]      = ollama_model_;
    body["messages"]   = messages;
    body["stream"]     = false;
    body["keep_alive"] = "30m";

    if (!ollama_options.empty()) {
        body["options"] = ollama_options;
    }

    int timeout = options.timeout_ms > 0 ? options.timeout_ms : 60000;
    std::string endpoint = url_ + "/api/chat";

    LOG_DEBUG("MMO_OLLAMA", "POST " + endpoint + " model=" + ollama_model_);

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
        LOG_ERROR("MMO_OLLAMA", result.error);
        return result;
    }

    auto j = nlohmann::json::parse(http_resp.text, nullptr, false);
    if (j.is_discarded()) {
        result.error = "Invalid JSON from " + endpoint + ": "
                       + http_resp.text.substr(0, 500);
        LOG_ERROR("MMO_OLLAMA", result.error);
        return result;
    }

    // Ollama /api/chat response: { "message": { "role": "assistant", "content": "..." }, ... }
    if (j.contains("message") && j["message"].contains("content")) {
        result.success = true;
        result.text    = j["message"]["content"].get<std::string>();
    } else {
        result.error = "Unexpected response format from " + endpoint;
        LOG_ERROR("MMO_OLLAMA", result.error);
        return result;
    }

    if (j.contains("eval_count")) {
        result.tokens_used = j["eval_count"].get<int>();
    }

    return result;
}

// =========================================================
// isAvailable — check if Ollama is reachable
// =========================================================

bool OllamaBackend::isAvailable() const {
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

std::string OllamaBackend::getBackendId() const {
    return "ollama:" + model_id_;
}

BackendType OllamaBackend::getBackendType() const {
    return BackendType::Ollama;
}

} // namespace GRIM::MMO
