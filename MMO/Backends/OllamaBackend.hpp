// OllamaBackend — IGenerationBackend for Ollama API
//
// Handles HTTP transport to the Ollama /api/chat endpoint.
// Used for sub-models (frozen bricks) in MMO mode.
//======================================================//
#pragma once

#include "../../ai/backends/IGenerationBackend.hpp"

#include <string>

namespace GRIM::MMO {

class OllamaBackend : public IGenerationBackend {
public:
    // url: base URL e.g. "http://127.0.0.1:11434"
    // model_id: registry model id
    // ollama_model: the Ollama model name e.g. "llama3.1:8b"
    OllamaBackend(const std::string& url,
                  const std::string& model_id,
                  const std::string& ollama_model);

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
    std::string ollama_model_;
};

} // namespace GRIM::MMO
