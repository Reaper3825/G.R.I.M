#pragma once
#include <string>
#include <future>
#include <functional>
#include <nlohmann/json.hpp>

// ------------------------------------------------------------
// Global AI state (persistent JSON containers)
// ------------------------------------------------------------

// Persistent long-term memory (saved to memory.json).
extern nlohmann::json longTermMemory;

// AI configuration (backend, model, URLs, etc., loaded from ai_config.json).
extern nlohmann::json aiConfig;

// ------------------------------------------------------------
// Runtime tunables (synced from aiConfig by loadAIConfig)
// ------------------------------------------------------------

// 🔹 Silence detection thresholds
extern double g_silenceThreshold;   // Minimum amplitude for speech detection
extern int    g_silenceTimeoutMs;   // Silence duration before timeout (ms)

// 🔹 Whisper tuning
extern std::string g_whisperLanguage; // Language code (e.g., "en")
extern int         g_whisperMaxTokens; // Max tokens Whisper will transcribe

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
/// @return "ollama", "localai", or "openai".
std::string resolveBackendURL();

// ------------------------------------------------------------
// Core AI call (async)
// ------------------------------------------------------------

/// Send a prompt to the configured AI backend asynchronously.
/// Returns a future so the caller can decide whether to block or poll.
std::future<std::string> callAIAsync(const std::string& prompt);

/// Warm up the AI backend at launch to avoid first-call delays.
void warmupAI();

// ------------------------------------------------------------
// Synchronous + streaming AI wrappers
// ------------------------------------------------------------

/// Blocking call to process text through AI and update memory.
/// Automatically speaks the reply aloud.
std::string ai_process(const std::string& input, nlohmann::json& memory);

/// Streaming call: feeds back partial chunks via callback.
/// Also streams chunks to audible output in real time.
void ai_process_stream(const std::string& input,
                       nlohmann::json& memory,
                       const std::function<void(const std::string&)>& onChunk);
