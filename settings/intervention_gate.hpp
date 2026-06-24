#pragma once
// =====================================================================
// Intervention gate — runtime on/off switches for GRIM's intent systems.
//
// Each "intervention" is a stage that processes a user's request BEFORE it
// reaches the language model (intent classification, NLP, grammar, command
// dispatch, RL suggestion, synonym rewriting, banter stripping, personality
// prefixing, router annotation). These switches let us bypass any stage —
// individually or all at once via the master switch — so the native
// grim-text model can be tested with and without the local scaffolding.
//
// Flags live in ai_config.json under "intent_systems" and are read from the
// live global `aiConfig`, so UI edits take effect on the next request with no
// restart. Every key defaults to ENABLED when absent, so older configs (and a
// missing block) behave exactly like today.
// =====================================================================
#include <string>
#include <nlohmann/json.hpp>

extern nlohmann::json aiConfig;

namespace Settings {

// Master switch. When false, ALL interventions are bypassed and the raw
// request is handed straight to the model.
inline bool interventionsMasterEnabled() {
    if (!aiConfig.contains("intent_systems") || !aiConfig["intent_systems"].is_object())
        return true; // default on — preserve current behavior
    return aiConfig["intent_systems"].value("master_enabled", true);
}

// Per-system switch. Returns false if the master switch is off (so every
// stage auto-bypasses), otherwise the system's own flag (default on).
// Side-effect-free: never inserts keys into aiConfig (no non-const operator[]).
inline bool interventionEnabled(const std::string& key) {
    auto it = aiConfig.find("intent_systems");
    if (it == aiConfig.end() || !it->is_object())
        return true; // no block → default on (current behavior)
    const nlohmann::json& systems = *it;
    if (!systems.value("master_enabled", true))
        return false; // master off → every stage bypasses
    return systems.value(key, true);
}

} // namespace Settings
