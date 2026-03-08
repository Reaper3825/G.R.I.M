// Multi-Model Orchestration (MMO) - Model Metadata Contracts
// Shared types for model descriptors, transport envelopes,
// and sub-model output contracts.
//======================================================//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Backend type — how the body connects to this model
// =========================================================
enum class BackendType : uint8_t {
    GrimTextServer = 0,   // grim_text_server.exe (HTTP, one process per model)
    LlamaCpp       = 1,   // llama.cpp server
    Ollama         = 2,   // Ollama API
    External       = 3    // arbitrary HTTP endpoint
};

// =========================================================
// ModelInfo — describes one model (router or sub-model)
//
// Router-only fields: lora_path, hard_copy_path
// Sub-models MUST leave those empty; ModelRegistry validates.
// =========================================================
struct ModelInfo {
    std::string id;
    std::string name;
    std::string version;
    std::string subject;
    std::string description;
    std::string model_path;         // weights file or directory
    BackendType backend_type   = BackendType::GrimTextServer;
    std::string url;                // host:port or full URL
    std::vector<std::string> subject_tags;
    float       usage_weight   = 0.0f;

    // Router-only — personalization bridge
    std::string lora_path;
    std::string hard_copy_path;

    // Estimated resource footprint (used by ResourceCoordinator)
    long        estimated_ram_mb  = 0;
    long        estimated_vram_mb = 0;
};

// =========================================================
// Transport envelope — body → model (route or synthesize step)
// Every request carries these fields.
// =========================================================
struct RequestEnvelope {
    uint32_t    schema_version = 1;
    std::string request_id;
    std::string session_id;
    std::string turn_id;
    std::string target_model_id;
    std::string task;
    std::string scope;
    std::string constraints;
    std::string output_schema;
    int         max_length     = 0;
    std::string payload;
};

// =========================================================
// Sub-model output envelope — model → body
// =========================================================
enum class ResponseStatus : uint8_t {
    Ok     = 0,
    Refuse = 1,
    Error  = 2
};

struct ResponseEnvelope {
    uint32_t       schema_version = 1;
    std::string    request_id;
    std::string    target_model_id;
    ResponseStatus status         = ResponseStatus::Error;
    std::string    result;         // when status == Ok
    std::string    refusal;        // when status == Refuse
    std::string    error;          // when status == Error
};

} // namespace GRIM::MMO