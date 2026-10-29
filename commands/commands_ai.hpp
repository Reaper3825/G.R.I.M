#pragma once
// Retained host/legacy handlers, not registered agent tools.
// Behavior and possible future uses: docs/RETIRED_AGENT_TOOLS.md.
#include "commands_core.hpp"

// =============================================================
// AI-related commands
// =============================================================

/**
 * @brief Show or change the active AI backend.
 * 
 * Usage:
 *   ai_backend             → shows current backend
 *   ai_backend <backend>   → sets backend (openai, ollama, localai, auto)
 */
CommandResult cmdAiBackend(const std::string& arg);

/**
 * @brief Reload NLP rules from disk.
 * 
 * Usage:
 *   reload_nlp
 */
CommandResult cmdReloadNlp(const std::string& arg);

/**
 * @brief Clear AI conversation context to start fresh.
 * 
 * Usage:
 *   clear_context
 */
CommandResult cmdClearContext(const std::string& arg);

/**
 * @brief General AI query (catch-all).
 * 
 * Usage:
 *   grim_ai <prompt>
 */
CommandResult cmdGrimAi(const std::string& arg);
