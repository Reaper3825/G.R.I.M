#pragma once
#include <string>
#include <unordered_map>
#include <nlohmann/json_fwd.hpp>


// ------------------------------------------------------------
// Aliases API (User + Auto + Fallback)
// ------------------------------------------------------------
// - User aliases: manually defined by the user.
// - Auto aliases: discovered by GRIM (PATH + Start Menu).
// - Fallback: fuzzy matching + system heuristics.
//
// Background refresh runs silently → grimLog() only.
// User-triggered refresh → console_history.push().
// ------------------------------------------------------------

namespace aliases {

    // Initialize system (called once at bootstrap).
    // Loads cached aliases.json immediately,
    // then spawns background refresh.
    void init();

    // Load aliases (user + auto) from JSON.
    // This does not trigger a refresh.
    void load();

    // Run async refresh (detached thread).
    // Silent → logs only with grimLog().
    void refreshAsync();

    // Run blocking refresh (user command).
    // Prints results into history.
    void refreshNow();

    // Resolve alias key into executable path.
    // Lookup order: [USER] → [AUTO] → [FALLBACK].
    // Returns empty string if not found.
    std::string resolve(const std::string& key);

    // Debug/Introspection API
    const nlohmann::json& getAll();   // returns current aliases.json in-memory
    std::string info(const std::string& key); // pretty metadata string

} // namespace aliases
