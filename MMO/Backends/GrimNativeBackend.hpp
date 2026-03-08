// GrimNativeBackend — IGenerationBackend for grim_text_server.exe
//
// Handles HTTP transport to grim-text's /api/chat, /api/route,
// /api/generate, and /api/synthesize endpoints.
//
// The grim-text server uses Ollama-compatible /api/chat format
// for basic generation, and MMO envelope format for route/
// generate/synthesize tasks.
//======================================================//
#pragma once

#include "../../ai/backends/IGenerationBackend.hpp"

#include <string>

namespace GRIM::MMO {

class GrimNativeBackend : public IGenerationBackend {
public:
    // url: base URL e.g. "http://127.0.0.1:11435"
    // model_id: registry model id e.g. "grim-text-router"
    GrimNativeBackend(const std::string& url, const std::string& model_id);

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

    // MMO envelope dispatch — POST full RequestEnvelope JSON to
    // the MMO endpoint and parse the ResponseEnvelope back.
    GenerationResult generateEnvelope(const GenerationOptions& options);
};

} // namespace GRIM::MMO
