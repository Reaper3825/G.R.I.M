#pragma once
#include <string>
#include <unordered_map>
#include <nlohmann/json_fwd.hpp>
#include "commands/commands_core.hpp"   // 🔹 For CommandResult

// ------------------------------------------------------------
// Aliases API (User + Auto + Fallback)
// ------------------------------------------------------------
// - User aliases: manually defined by the user.
// - Auto aliases: discovered by GRIM (D: drive + Start Menu).
// - Fallback: fuzzy matching + heuristics.
//
// Background refresh runs silently → grimLog() only.
// User-triggered refresh → returns CommandResult.
// 
// 🔹 Thread-safety:
//   - All public APIs are thread-safe.
//   - refreshAsync/refreshNow are guarded by an internal
//     reentrancy lock (atomic flag), so scans never overlap.
// ------------------------------------------------------------

namespace aliases {

    void init();        // Initialize system (bootstrap)
    void load();        // Load aliases (user + auto) from JSON
    void refreshAsync();// Run async refresh (silent/log-only)

    // 🔹 Now returns CommandResult for unified response flow
    CommandResult refreshNow();

    // Resolve alias key into executable path.
    // Lookup order: [USER] → [AUTO] → [FALLBACK fuzzy].
    std::string resolve(const std::string& key);

    // Debug/Introspection API
    const nlohmann::json& getAll();
    std::string info(const std::string& key);

} // namespace aliases
