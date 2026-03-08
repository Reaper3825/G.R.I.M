// Multi-Model Orchestration (MMO) - SessionContextManager
// Session-scoped interaction state authority.
//
// Replaces the fragmented context layer:
//   - ContextManager static state (recentContext, pendingIntent)
//   - commands_feedback.cpp globals (g_pendingClarifyCmd, g_pendingFeedbackCmd)
//   - hardcoded referent resolution in commands_core.cpp
//
// One canonical owner of live conversational/task context per session.
// Session state is keyed, not process-global.
//======================================================//
#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory/unified_memory.hpp"
#include "memory/context_snapshot.hpp"

namespace GRIM::MMO {

// =========================================================
// TurnRecord — append-only record of a single interaction turn
// =========================================================
struct TurnRecord {
    std::string session_id;
    std::string turn_id;
    std::chrono::steady_clock::time_point timestamp;

    // User input
    struct {
        std::string raw;
        std::string normalized;
    } user_input;

    // NLP annotation summary (compact — not the full NlpAnnotation)
    std::string nlp_summary;

    // Tags produced by NLP / router
    std::vector<std::string> router_tags;
    std::vector<std::string> memory_tags;
    std::vector<std::string> risk_tags;

    // Outcome tracking
    std::string selected_route;      // sub-model id or "direct"
    std::string proposed_command;    // tool_id if action was proposed
    std::string final_outcome;       // result summary
};

// =========================================================
// ReferentBinding — tracked entity for pronoun resolution
// =========================================================
struct ReferentBinding {
    std::string canonical_id;        // unique entity identifier
    std::string value;               // display / lookup value
    std::string entity_type;         // "app", "file", "path", "url", "person", "window", etc.
    std::string source_turn_id;      // which turn introduced this referent
    float       confidence = 1.0f;
    std::chrono::steady_clock::time_point created_at;
    int         ttl_seconds = 300;   // default 5 minute expiry
};

// =========================================================
// PendingInteraction — unified pending state
// (replaces g_pendingClarifyCmd, g_pendingFeedbackCmd, PendingIntent)
// =========================================================
enum class PendingKind : uint8_t {
    MissingSlot,     // command needs a parameter
    Clarification,   // ambiguous input needs disambiguation
    Confirmation,    // risky action needs user approval
    Correction,      // user corrected a previous action
    FollowUp         // multi-turn continuation
};

struct PendingInteraction {
    PendingKind kind;
    std::string original_command;    // what was proposed
    std::string missing_field;       // which slot/param is missing (MissingSlot)
    std::string prompt_shown;        // what was asked of the user
    std::string turn_id;             // which turn created this
    std::chrono::steady_clock::time_point created_at;
    int         expiry_seconds = 120;
};

// =========================================================
// ActionEpisode — tracks a proposed action through its lifecycle
// =========================================================
struct ActionEpisode {
    std::string tool_id;
    std::string proposed_args;       // serialized arguments
    float       risk     = 0.0f;
    float       confidence = 1.0f;
    bool        confirmation_requested = false;
    bool        user_rejected   = false;
    std::string correction_text;     // non-empty if user corrected
    std::string accepted_action;     // final action after corrections
    std::string execution_result;    // outcome
    std::string turn_id;
};

// =========================================================
// VisualContext — split physical vs digital
// =========================================================
struct VisualContext {
    // Digital: on-screen / desktop / monitor state
    struct DigitalVisual {
        std::string active_window;
        std::string active_process;
        std::string ocr_text;
        std::string scene_classification;
        std::string monitor_id;
    } digital;

    // Physical: real-world / camera semantic input (future)
    struct PhysicalVisual {
        std::string scene_summary;
        std::vector<std::string> detected_objects;
    } physical;
};

// =========================================================
// ContextSnapshotV2 — the rich projection consumed by
// router, FastClassifier, memory retrieval, Training Wheels
// =========================================================
struct ContextSnapshotV2 {
    std::string session_id;
    std::string turn_id;
    std::string active_task_id;

    // Recent turns (summaries, not full records)
    std::vector<std::string> recent_turn_summaries;

    // NLP state
    std::string latest_nlp_summary;
    std::vector<std::string> utterance_priors;  // e.g. "command:0.8", "question:0.2"

    // Active referents
    std::vector<ReferentBinding> active_referents;

    // Pending interaction
    std::optional<PendingInteraction> pending;

    // Recent action outcomes
    std::vector<ActionEpisode> recent_episodes;

    // Memory retrieval breadcrumbs
    std::vector<std::string> memory_breadcrumbs;

    // Visual context (split physical / digital)
    VisualContext visual_context;

    // Mood and resource summary
    std::string current_mood;
    std::string resource_pressure;  // "healthy", "pressured", "critical"

    // Risk summary for current turn
    std::vector<std::string> risk_tags;

    // Legacy compatibility projection
    std::string lastNlpCategory;
    int consecutiveCommands = 0;
};

// =========================================================
// SessionContextManager
//
// Usage:
//   auto& ctx = SessionContextManager::instance();
//   ctx.beginTurn("session-1", "turn-42", "hello world", "hello world");
//   ctx.addReferent({"app:notepad", "Notepad", "app", "turn-42"});
//   auto snap = ctx.snapshot("session-1");
//
// Thread-safe: all operations serialized under session mutex.
// =========================================================
class SessionContextManager {
public:
    static SessionContextManager& instance();

    // ─── Turn lifecycle ───────────────────────────────────

    // Begin a new turn. Creates a TurnRecord and appends it.
    void beginTurn(const std::string& session_id,
                   const std::string& turn_id,
                   const std::string& raw_input,
                   const std::string& normalized_input);

    // Attach NLP annotation summary to the current turn.
    void setNlpSummary(const std::string& session_id,
                       const std::string& nlp_summary);

    // Attach routing/memory/risk tags to the current turn.
    void setTurnTags(const std::string& session_id,
                     const std::vector<std::string>& router_tags,
                     const std::vector<std::string>& memory_tags,
                     const std::vector<std::string>& risk_tags);

    // Record the outcome of a turn (route + command + result).
    void recordOutcome(const std::string& session_id,
                       const std::string& selected_route,
                       const std::string& proposed_command,
                       const std::string& final_outcome);

    // ─── Referent tracking ────────────────────────────────

    // Add or update a referent binding.
    void addReferent(const std::string& session_id,
                     const ReferentBinding& binding);

    // Resolve a pronoun/reference like "it", "that file", "same folder".
    std::optional<ReferentBinding> resolveReference(
        const std::string& session_id,
        const std::string& reference_text) const;

    // ─── Pending interaction ──────────────────────────────

    // Set pending interaction state (replaces g_pendingClarifyCmd etc.)
    void setPending(const std::string& session_id,
                    const PendingInteraction& interaction);

    // Get current pending interaction if any.
    std::optional<PendingInteraction> getPending(
        const std::string& session_id) const;

    // Clear pending interaction.
    void clearPending(const std::string& session_id);

    // ─── Action episodes ──────────────────────────────────

    // Record a proposed action.
    void recordProposal(const std::string& session_id,
                        const ActionEpisode& episode);

    // Update the latest episode with user response (accept/reject/correct).
    void recordUserResponse(const std::string& session_id,
                            bool rejected,
                            const std::string& correction_text);

    // Update the latest episode with execution result.
    void recordExecutionResult(const std::string& session_id,
                               const std::string& result);

    // ─── Visual context ───────────────────────────────────

    // Update digital visual context (from perception system).
    void updateDigitalVisual(const std::string& session_id,
                             const VisualContext::DigitalVisual& digital);

    // Update physical visual context (from camera/sensors).
    void updatePhysicalVisual(const std::string& session_id,
                              const VisualContext::PhysicalVisual& physical);

    // ─── Snapshot ─────────────────────────────────────────

    // Build the rich context snapshot for router / retrieval / policy.
    ContextSnapshotV2 snapshot(const std::string& session_id) const;

    // ─── Maintenance ──────────────────────────────────────

    // Expire old referents, pending interactions, and trim turn history.
    void tick(const std::string& session_id);

    // Destroy a session's state entirely.
    void destroySession(const std::string& session_id);

    // Set mood (from PersonalityManager bridge).
    void setMood(const std::string& session_id, const std::string& mood);

    // Set resource pressure (from ResourceSignal bridge).
    void setResourcePressure(const std::string& session_id,
                             const std::string& pressure);

    // ─── Voice / multi-command flags ──────────────────────

    // Track whether the current command came from voice input.
    void setVoiceCommand(const std::string& session_id, bool is_voice);
    bool isVoiceCommand(const std::string& session_id) const;

    // Track whether we are inside a multi-command batch.
    void setMultiCommandContext(const std::string& session_id, bool is_multi);
    bool isMultiCommandContext(const std::string& session_id) const;

    // ─── Usage tracking ───────────────────────────────────

    // Increment usage counter for a command/category.
    void recordUsage(const std::string& session_id, const std::string& category);
    int usageCount(const std::string& session_id, const std::string& category) const;

    // ─── Context memory (replaces ContextManager statics) ─

    // Store a context object for short-term recall.
    void rememberContextObject(const std::string& session_id,
                               const UnifiedMemoryObject& obj);

    // Recall most recent context object matching a TypeTag string.
    std::optional<UnifiedMemoryObject> recallContextByType(
        const std::string& session_id,
        const std::string& type_tag) const;

    // Recall most recent context object matching an intent string.
    std::optional<UnifiedMemoryObject> recallContextByIntent(
        const std::string& session_id,
        const std::string& intent_tag) const;

    // Remove context objects older than `seconds`.
    void decayOldContext(const std::string& session_id, int seconds);

    // ─── Conversation history (replaces g_conversationHistory) ─

    // A single chat message in the conversation history.
    struct ChatMessage {
        std::string role;    // "system" | "user" | "assistant"
        std::string content;
    };

    // Add a message to the session's conversation history.
    void addMessage(const std::string& session_id,
                    const std::string& role,
                    const std::string& content);

    // Set the system prompt (inserted at position 0, only once).
    void setSystemPrompt(const std::string& session_id,
                         const std::string& prompt);

    // Get the current conversation history for building API requests.
    std::vector<ChatMessage> getMessages(const std::string& session_id) const;

    // Clear conversation history (e.g. on model switch).
    void clearHistory(const std::string& session_id);

    // Trim history to at most `max_messages` (preserving system prompt).
    void trimHistory(const std::string& session_id, size_t max_messages);

    // ─── Legacy V1 snapshot projection ────────────────────

    // Build a ContextSnapshot (legacy V1 type) for callers that
    // still consume the old format (IntentGate, FastClassifier).
    GRIM::ContextSnapshot legacySnapshot(const std::string& session_id) const;

private:
    SessionContextManager() = default;

    // Per-session state bucket
    struct SessionState {
        std::string session_id;
        std::vector<TurnRecord> turns;
        std::vector<ReferentBinding> referents;
        std::optional<PendingInteraction> pending;
        std::vector<ActionEpisode> episodes;
        VisualContext visual;
        std::string mood;
        std::string resource_pressure;
        int consecutive_commands = 0;
        std::string last_nlp_category;

        // Voice / multi-command session flags
        bool is_voice_command = false;
        bool is_multi_command_context = false;

        // Usage counters per category
        std::unordered_map<std::string, int> usage_counts;

        // Short-term context objects for recall
        std::vector<UnifiedMemoryObject> recent_context;

        // Conversation history for multi-turn API requests
        std::vector<ChatMessage> conversation_history;
        bool system_prompt_set = false;
    };

    SessionState& getOrCreate(const std::string& session_id);
    const SessionState* get(const std::string& session_id) const;

    static constexpr int kMaxTurnsPerSession = 100;
    static constexpr int kMaxEpisodes = 20;
    static constexpr int kMaxReferents = 50;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, SessionState> sessions_;
};

} // namespace GRIM::MMO
