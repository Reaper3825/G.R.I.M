#pragma once
#include <string>
#include <future>
#include <functional>
#include <nlohmann/json_fwd.hpp>
#include "commands/commands_core.hpp"

// ------------------------------------------------------------
// Global AI state (persistent JSON containers)
// ------------------------------------------------------------

// Persistent long-term memory (saved to memory.json).
extern nlohmann::json longTermMemory;

// AI configuration (backend, model, URLs, etc., loaded by bootstrap_config).
extern nlohmann::json aiConfig;

// ------------------------------------------------------------
// Runtime tunables (synced from aiConfig at bootstrap)
// ------------------------------------------------------------
extern double g_silenceThreshold;   // Minimum amplitude for speech detection
extern int    g_silenceTimeoutMs;  // Silence duration before timeout (ms)
extern std::string g_whisperLanguage; 
extern int         g_whisperMaxTokens;

// ------------------------------------------------------------
// Persistence functions
// ------------------------------------------------------------
void loadMemory();   // Load memory.json into longTermMemory
void saveMemory();   // Save longTermMemory back to memory.json

// ------------------------------------------------------------
// Voice helpers (store in longTermMemory["voice"])
// ------------------------------------------------------------
void rememberCorrection(const std::string& wrong, const std::string& right);
void rememberShortcut(const std::string& phrase, const std::string& command);
void incrementUsageCount(const std::string& command);
void setLastCommand(const std::string& command);

// ------------------------------------------------------------
// AI backend resolver
// ------------------------------------------------------------
std::string resolveBackendURL();

// ------------------------------------------------------------
// Core AI calls
// ------------------------------------------------------------
std::future<std::string> callAIAsync(const std::string& prompt);

// Warm up the AI backend at launch to avoid first-call delays.
void warmupAI();

// ------------------------------------------------------------
// Synchronous + streaming AI wrappers
// ------------------------------------------------------------
CommandResult ai_process(const std::string& input);

void ai_process_stream(const std::string& input,
                       nlohmann::json& memory,
                       const std::function<void(const std::string&)>& onChunk);
