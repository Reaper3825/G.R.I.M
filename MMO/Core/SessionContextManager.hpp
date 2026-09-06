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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory/unified_memory.hpp"
#include "console_history.hpp"
#include "DataCollection/reasoning_state.hpp"

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
// VisualContext — split physical vs digital
// =========================================================
struct VisualContext {
    // Digital: on-screen / desktop / monitor state
    struct DigitalVisual {
        std::string active_window;
        std::string active_process;
        std::string ocr_text;
        std::vector<std::string> ocr_regions;
        std::vector<std::string> ui_elements;
        std::string scene_classification;
        std::string monitor_id;
        std::string source_device_id;
        std::string source_platform;
        std::string source_transport;
        std::string capture_status;
        std::string capture_error;
        std::string capture_backend;
        std::string ocr_status;
        std::string ocr_error;
        std::string ocr_provider;
        std::string automation_status;
        std::string automation_error;
        std::string automation_provider;
        std::string automation_target_window;
        std::string preferred_grounding_source;
        float       ocr_mean_confidence = 0.0f;
        bool        automation_target_matches_capture = false;
        bool        automation_target_changed = false;
        uint64_t    provenance_frame_counter = 0;
        uint64_t    primitive_provenance_frame_counter = 0;
        uint64_t    capture_wall_ns = 0;
    } digital;

    // Physical: real-world / camera semantic input.
    // Populated each frame by PhysicalWorldStateContextProjector from the
    // latest PhysicalWorldStateBus snapshot. All fields are LM-readable
    // strings — no pixel coordinates, no boxes, no per-frame floats — so
    // the router never sees raw geometry tokens. provenance_frame_counter
    // is for diagnostics only; consumers MUST NOT use it for routing.
    struct PhysicalVisual {
        std::string scene_summary;                       // 1-line headline ("3 visible, 1 occluded")
        std::vector<std::string> detected_objects;       // class labels with counts ("cup x2, person x1")
        std::vector<std::string> entities_in_focus;      // "cup#7 on desk, 0.4m, holds 'COFFEE'"
        std::vector<std::string> spatial_relations_top_k;// "cup#7 Contains lid#9 (0.81)"
        std::vector<std::string> active_alerts;          // "path_blocked: chair#4 (0.92)"
        uint64_t provenance_frame_counter = 0;
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

    // Memory retrieval breadcrumbs
    std::vector<std::string> memory_breadcrumbs;

    // Visual context (split physical / digital)
    VisualContext visual_context;

    // Mood and resource summary
    std::string current_mood;
    std::string resource_pressure;  // "healthy", "pressured", "critical"

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
    // State persists until replaced/cleared; console submission copies a per-request snapshot.
    void setReasoningState(const std::string& session_id, std::optional<GRIM::ReasoningState> state);
    std::optional<GRIM::ReasoningState> getReasoningState(const std::string& session_id) const;

    // ─── Turn lifecycle ───────────────────────────────────

    // Begin a new turn. Creates a TurnRecord and appends it.
    void beginTurn(const std::string& session_id,
                   const std::string& turn_id,
                   const std::string& raw_input,
                   const std::string& normalized_input);

    // Attach NLP annotation summary to the current turn.
    void setNlpSummary(const std::string& session_id,
                       const std::string& nlp_summary);

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

    // ─── Context memory (replaces ContextManager statics) ─

    // Store a context object for short-term recall.
    void rememberContextObject(const std::string& session_id,
                               const UnifiedMemoryObject& obj);

    // Recall most recent context object matching a TypeTag string.
    std::optional<UnifiedMemoryObject> recallContextByType(
        const std::string& session_id,
        const std::string& type_tag) const;

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

    // Session-owned display history used by console surfaces.
    ConsoleHistory& displayHistory(const std::string& session_id);

private:
    SessionContextManager() = default;

    // Per-session state bucket
    struct SessionState {
        std::optional<GRIM::ReasoningState> reasoning_state;
        std::string session_id;
        std::vector<TurnRecord> turns;
        std::vector<ReferentBinding> referents;
        std::optional<PendingInteraction> pending;
        VisualContext visual;
        std::string mood;
        std::string resource_pressure;
        int consecutive_commands = 0;
        std::string last_nlp_category;

        // Short-term context objects for recall
        std::vector<UnifiedMemoryObject> recent_context;

        // Conversation history for multi-turn API requests
        std::vector<ChatMessage> conversation_history;
        bool system_prompt_set = false;

        std::unique_ptr<ConsoleHistory> display_history =
            std::make_unique<ConsoleHistory>();
    };

    SessionState& getOrCreate(const std::string& session_id);
    const SessionState* get(const std::string& session_id) const;

    static constexpr int kMaxTurnsPerSession = 100;
    static constexpr int kMaxReferents = 50;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, SessionState> sessions_;
};

} // namespace GRIM::MMO
