// LlamaCppBackend — IGenerationBackend for llama.cpp server
//
// Handles HTTP transport to llama.cpp's OpenAI-compatible
// /v1/chat/completions endpoint.
//======================================================//
#pragma once

#include "../../ai/backends/IGenerationBackend.hpp"

#include <string>

namespace GRIM::MMO {

class LlamaCppBackend : public IGenerationBackend {
public:
    // url: base URL e.g. "http://127.0.0.1:8080"
    // model_id: registry model id e.g. "medical-7b"
    // model_name: model name to send in requests (may differ from model_id)
    LlamaCppBackend(const std::string& url,
                    const std::string& model_id,
                    const std::string& model_name);

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
    std::string model_name_;
};

} // namespace GRIM::MMO
