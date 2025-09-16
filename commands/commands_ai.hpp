#pragma once
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
 * @brief General AI query (catch-all).
 * 
 * Usage:
 *   grim_ai <prompt>
 */
CommandResult cmdGrimAi(const std::string& arg);

// =============================================================
// App / Web commands
// =============================================================

/**
 * @brief Open a local application using an alias.
 * 
 * Usage:
 *   open_app <application>
 */
CommandResult cmdOpenApp(const std::string& arg);

/**
 * @brief Search the web with the default browser.
 * 
 * Usage:
 *   search_web <query>
 */
CommandResult cmdSearchWeb(const std::string& arg);
