// Multi-Model Orchestration (MMO) - MemoryFacade
// Unified retrieval/write surface over the split memory subsystem.
//
// Wraps:
//   - UnifiedMemoryStorage  (FlatBuffer-backed storage with indexes)
//   - ContextManager         (session-scoped interaction state)
//   - MemoryRouter           (routing classification)
//
// The orchestrator and other MMO consumers use ONLY this facade
// for memory operations. They never touch storage or context
// classes directly.
//
// This is an adapter — existing non-MMO consumers keep using
// storage/context directly until they are migrated.
//======================================================//
#pragma once

#include "memory/unified_memory.hpp"
#include "SessionContextManager.hpp"
#include "memory/memory_router.hpp"

#include <mutex>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// MemoryRetrievalResult — returned from retrieveForPrompt()
//
// Contains everything the orchestrator needs to compose
// context for the router: ranked memories, session snapshot,
// and retrieval breadcrumbs (what was searched and why).
// =========================================================
struct MemoryRetrievalResult {
    // Ranked memories relevant to the prompt (most relevant first).
    std::vector<UnifiedMemoryObject> memories;

    // Current session state (mood, recent commands, intents, NLP context).
    ContextSnapshot context;

    // Retrieval breadcrumbs: what queries were used and how many
    // results each produced. Useful for diagnostics and router
    // confidence estimation.
    struct Breadcrumb {
        std::string query;
        int         results_found = 0;
    };
    std::vector<Breadcrumb> breadcrumbs;

    // True if the retrieval produced at least one relevant memory.
    bool has_memories() const { return !memories.empty(); }
};

// =========================================================
// MemoryFacade
//
// Usage:
//   MemoryFacade facade(storage);
//   auto retrieval = facade.retrieveForPrompt("explain photosynthesis");
//   // Build router payload from retrieval.memories + retrieval.context
//
// Thread-safe: internal mutex protects all operations.
// =========================================================
class MemoryFacade {
public:
    // Construct with a reference to the canonical storage engine.
    // The storage must outlive the facade.
    explicit MemoryFacade(UnifiedMemoryStorage& storage);

    // ─── Retrieval ────────────────────────────────────────

    // Primary retrieval entry point for the orchestrator.
    // Searches storage by text, tags, and type; merges with
    // session context snapshot; returns a ranked result set.
    MemoryRetrievalResult retrieveForPrompt(
        const std::string& prompt,
        int max_memories = 10) const;

    // Direct search passthrough.
    std::vector<UnifiedMemoryObject> search(
        const std::string& query,
        int max_results = 10) const;

    // Tag-based retrieval.
    std::vector<UnifiedMemoryObject> getByTag(
        const std::string& tag) const;

    // Multi-tag retrieval.
    std::vector<UnifiedMemoryObject> getByTags(
        const std::vector<std::string>& tags,
        bool match_all = false) const;

    // ─── Recording ────────────────────────────────────────

    // Record a completed interaction turn into memory.
    // Routes through MemoryRouter to decide short/long-term placement.
    void recordInteraction(const UnifiedMemoryObject& obj);

    // Store a memory directly into long-term storage.
    void storeLongTerm(const UnifiedMemoryObject& obj);

    // Store a memory directly into short-term storage.
    void storeShortTerm(const UnifiedMemoryObject& obj);

    // ─── Session context passthrough ──────────────────────

    // Get a snapshot of the current session state.
    ContextSnapshot getContextSnapshot() const;

    // Record a context memory for session-scoped recall.
    void rememberContext(const UnifiedMemoryObject& obj);

    // Decay old session context entries.
    void decayContext(int seconds = 180);

    // ─── Maintenance ──────────────────────────────────────

    // Flush all pending writes to disk.
    void flush();

    // Compact storage (removes low-confidence entries).
    void compact();

private:
    UnifiedMemoryStorage& storage_;
    mutable std::mutex    mutex_;
};

} // namespace GRIM::MMO
