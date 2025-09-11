#pragma once
#include <string>
#include <future>
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Global AI state
// ------------------------------------------------------------

// Persistent long-term memory (key/value store, saved to memory.json).
extern nlohmann::json longTermMemory;

// AI configuration (backend, model, URLs, etc., loaded from ai_config.json).
extern nlohmann::json aiConfig;

// ------------------------------------------------------------
// Persistence functions
// ------------------------------------------------------------

/// Load memory.json into longTermMemory.
/// Creates a new file if missing or invalid.
void loadMemory();

/// Save longTermMemory back to memory.json.
void saveMemory();

// ------------------------------------------------------------
// AI configuration
// ------------------------------------------------------------

/// Load AI backend configuration from JSON file.
/// Falls back to defaults if missing or invalid.
void loadAIConfig(const std::string& path);

/// Resolve the backend URL depending on platform and config.
/// @return Full backend URL (e.g., Ollama or LocalAI endpoint).
std::string resolveBackendURL();

// ------------------------------------------------------------
// Core AI call (async)
// ------------------------------------------------------------

/// Send a prompt to the configured AI backend asynchronously.
/// Returns a future so the caller can decide whether to block or poll.
/// @param prompt The user input to send.
/// @return Future string containing the AI-generated reply or error message.
std::future<std::string> callAIAsync(const std::string& prompt);

/// Warm up the AI backend at launch to avoid first-call delays.
void warmupAI();
