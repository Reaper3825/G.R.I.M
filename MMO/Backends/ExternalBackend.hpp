// ExternalBackend — IGenerationBackend for arbitrary HTTP endpoints
//
// Generic HTTP POST backend for external model servers.
// Supports two response formats:
//   1. OpenAI-compatible: { "choices": [{ "message": { "content": "..." } }] }
//   2. Raw text: response body is treated as generated text
//======================================================//
#pragma once

#include "../../ai/backends/IGenerationBackend.hpp"

#include <string>

namespace GRIM::MMO {

class ExternalBackend : public IGenerationBackend {
public:
    // url: base URL e.g. "http://192.168.1.10:5000"
    // model_id: registry model id
    // endpoint_path: path to POST to, e.g. "/v1/chat/completions" or "/generate"
    ExternalBackend(const std::string& url,
                    const std::string& model_id,
                    const std::string& endpoint_path);

    GenerationResult generate(
        const std::string& prompt,
        const GenerationOptions& options) override;

    GenerationResult generateWithHistory(
        const std::string& prompt,
        const std::vector<HistoryEntry>& history,
        const GenerationOptions& options) override;

    bool isAvailable() const override;

    std::string getBackendId() const override;

    BackendType getBackendType() const override;

private:
    std::string url_;
    std::string model_id_;
    std::string endpoint_path_;
};

} // namespace GRIM::MMO
