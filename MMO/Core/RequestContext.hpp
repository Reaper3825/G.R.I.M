// MMO RequestContext — per-request state carried through orchestration
//
// Every orchestrator call receives a RequestContext.  It binds:
//   request_id, session_id, turn_id   — envelope correlation
//   deadline                          — absolute timeout
//   session-scoped history handle     — for memory/context queries
//
// This is a lightweight value type, NOT a managed session.
// Session lifecycle is owned by the caller (context_manager /
// body main loop); the orchestrator only borrows this context
// for the duration of a single generate() call.
//======================================================//
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// TurnSummary — minimal record of a single conversation turn
// used when building router metadata.  The full TurnRecord
// lives in the session context authority; this is a read-only
// projection for cross-boundary transfer.
// =========================================================
struct TurnSummary {
    std::string turn_id;
    std::string role;       // "user" | "assistant" | "system"
    std::string text;       // truncated if needed
    int         token_count = 0;
};

// =========================================================
// RequestContext — immutable per-request snapshot
//
// Constructed by the caller before calling Orchestrator::generate().
// Orchestrator passes pieces to the envelope builders as needed.
// =========================================================
struct RequestContext {
    // ── Correlation IDs ──
    std::string request_id;
    std::string session_id;
    std::string turn_id;

    // ── Prompt / content ──
    std::string prompt;
    std::string system_prompt;

    // ── Precomposed metadata (from RouterMetadataBuilder) ──
    std::string metadata_json;

    // ── Timeouts ──
    // Absolute deadline for the entire orchestration cycle.
    // If empty (default), the orchestrator uses per-step
    // timeouts from OrchestratorConfig.
    std::chrono::steady_clock::time_point deadline{};

    // ── Session-scoped history ──
    // Recent turns for context window.  Populated by the caller
    // from ContextManager / session authority.
    std::vector<TurnSummary> recent_turns;

    // ── Tool-surface fingerprint ──
    // Registry version at the time the request was constructed.
    // Allows the orchestrator to include/exclude tool summary
    // regeneration based on whether the registry has changed.
    uint64_t tool_registry_version = 0;

    // ── Helpers ──

    // True if a deadline has been set.
    bool hasDeadline() const {
        return deadline != std::chrono::steady_clock::time_point{};
    }

    // Milliseconds remaining until deadline, or -1 if no deadline.
    int64_t remainingMs() const {
        if (!hasDeadline()) return -1;
        auto now = std::chrono::steady_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        return ms > 0 ? ms : 0;
    }
};

} // namespace GRIM::MMO
