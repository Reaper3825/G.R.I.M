#include "SessionContextManager.hpp"

#include <algorithm>
#include <ctime>
#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Singleton
// =========================================================

SessionContextManager& SessionContextManager::instance() {
    static SessionContextManager s;
    return s;
}

// =========================================================
// Internal helpers
// =========================================================

SessionContextManager::SessionState&
SessionContextManager::getOrCreate(const std::string& session_id) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        auto [ins, ok] = sessions_.emplace(session_id, SessionState{});
        ins->second.session_id = session_id;
        return ins->second;
    }
    return it->second;
}

const SessionContextManager::SessionState*
SessionContextManager::get(const std::string& session_id) const {
    auto it = sessions_.find(session_id);
    return (it != sessions_.end()) ? &it->second : nullptr;
}

// =========================================================
// Turn lifecycle
// =========================================================

void SessionContextManager::beginTurn(
    const std::string& session_id,
    const std::string& turn_id,
    const std::string& raw_input,
    const std::string& normalized_input)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);

    TurnRecord rec;
    rec.session_id = session_id;
    rec.turn_id = turn_id;
    rec.timestamp = std::chrono::steady_clock::now();
    rec.user_input.raw = raw_input;
    rec.user_input.normalized = normalized_input;
    s.turns.push_back(std::move(rec));

    // Trim oldest turns if over limit
    if (static_cast<int>(s.turns.size()) > kMaxTurnsPerSession) {
        s.turns.erase(s.turns.begin(),
                      s.turns.begin() + (static_cast<int>(s.turns.size()) - kMaxTurnsPerSession));
    }
}

void SessionContextManager::setNlpSummary(
    const std::string& session_id,
    const std::string& nlp_summary)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    if (s.turns.empty()) return;
    s.turns.back().nlp_summary = nlp_summary;
    s.last_nlp_category = nlp_summary;
}

// =========================================================
// Referent tracking
// =========================================================

void SessionContextManager::addReferent(
    const std::string& session_id,
    const ReferentBinding& binding)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);

    // Update existing referent with same canonical_id
    for (auto& r : s.referents) {
        if (r.canonical_id == binding.canonical_id) {
            r = binding;
            r.created_at = std::chrono::steady_clock::now();
            return;
        }
    }

    // Add new, evict oldest if over limit
    if (static_cast<int>(s.referents.size()) >= kMaxReferents) {
        s.referents.erase(s.referents.begin());
    }
    ReferentBinding b = binding;
    b.created_at = std::chrono::steady_clock::now();
    s.referents.push_back(std::move(b));
}

std::optional<ReferentBinding> SessionContextManager::resolveReference(
    const std::string& session_id,
    const std::string& reference_text) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* s = get(session_id);
    if (!s || s->referents.empty()) return std::nullopt;

    // Map common pronouns/deictics to entity types
    std::string target_type;
    if (reference_text == "it" || reference_text == "that") {
        // Most recent referent of any type
    } else if (reference_text == "that app" || reference_text == "the app") {
        target_type = "app";
    } else if (reference_text == "that file" || reference_text == "the file") {
        target_type = "file";
    } else if (reference_text == "same folder" || reference_text == "that folder") {
        target_type = "path";
    } else if (reference_text == "that window" || reference_text == "the window") {
        target_type = "window";
    } else if (reference_text == "that url" || reference_text == "the url" ||
               reference_text == "that link" || reference_text == "the link") {
        target_type = "url";
    }

    auto now = std::chrono::steady_clock::now();

    // Search backwards (most recent first)
    for (auto it = s->referents.rbegin(); it != s->referents.rend(); ++it) {
        // Check expiry
        auto age_s = std::chrono::duration_cast<std::chrono::seconds>(
            now - it->created_at).count();
        if (age_s > it->ttl_seconds) continue;

        if (target_type.empty() || it->entity_type == target_type) {
            return *it;
        }
    }

    return std::nullopt;
}

// =========================================================
// Pending interaction
// =========================================================

void SessionContextManager::setPending(
    const std::string& session_id,
    const PendingInteraction& interaction)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    PendingInteraction p = interaction;
    p.created_at = std::chrono::steady_clock::now();
    s.pending = std::move(p);
}

std::optional<PendingInteraction> SessionContextManager::getPending(
    const std::string& session_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* s = get(session_id);
    if (!s) return std::nullopt;
    return s->pending;
}

void SessionContextManager::clearPending(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.pending.reset();
}

// =========================================================
// Visual context
// =========================================================

void SessionContextManager::updateDigitalVisual(
    const std::string& session_id,
    const VisualContext::DigitalVisual& digital)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.visual.digital = digital;
}

void SessionContextManager::updatePhysicalVisual(
    const std::string& session_id,
    const VisualContext::PhysicalVisual& physical)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.visual.physical = physical;
}

// =========================================================
// Snapshot
// =========================================================

ContextSnapshotV2 SessionContextManager::snapshot(
    const std::string& session_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    ContextSnapshotV2 snap;
    snap.session_id = session_id;

    const auto* s = get(session_id);
    if (!s) return snap;

    // Turn ID — latest turn
    if (!s->turns.empty()) {
        snap.turn_id = s->turns.back().turn_id;
        snap.latest_nlp_summary = s->turns.back().nlp_summary;
    }

    // Recent turn summaries (last 10)
    {
        int start = std::max(0, static_cast<int>(s->turns.size()) - 10);
        for (int i = start; i < static_cast<int>(s->turns.size()); ++i) {
            const auto& t = s->turns[i];
            std::string summary = t.user_input.raw;
            snap.recent_turn_summaries.push_back(std::move(summary));
        }
    }

    // Active referents (filter expired)
    {
        auto now = std::chrono::steady_clock::now();
        for (const auto& r : s->referents) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - r.created_at).count();
            if (age <= r.ttl_seconds) {
                snap.active_referents.push_back(r);
            }
        }
    }

    // Pending interaction
    snap.pending = s->pending;

    // Visual context
    snap.visual_context = s->visual;

    // Mood and pressure
    snap.current_mood = s->mood;
    snap.resource_pressure = s->resource_pressure;

    // Legacy compat fields
    snap.lastNlpCategory = s->last_nlp_category;
    snap.consecutiveCommands = s->consecutive_commands;

    return snap;
}

// =========================================================
// Maintenance
// =========================================================

void SessionContextManager::tick(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* s = const_cast<SessionState*>(get(session_id));
    if (!s) return;

    auto now = std::chrono::steady_clock::now();

    // Expire old referents
    s->referents.erase(
        std::remove_if(s->referents.begin(), s->referents.end(),
            [&](const ReferentBinding& r) {
                auto age = std::chrono::duration_cast<std::chrono::seconds>(
                    now - r.created_at).count();
                return age > r.ttl_seconds;
            }),
        s->referents.end());

    // Expire pending interaction
    if (s->pending.has_value()) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - s->pending->created_at).count();
        if (age > s->pending->expiry_seconds) {
            s->pending.reset();
        }
    }
}

void SessionContextManager::destroySession(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(session_id);
}

void SessionContextManager::setMood(
    const std::string& session_id, const std::string& mood) {
    std::lock_guard<std::mutex> lock(mutex_);
    getOrCreate(session_id).mood = mood;
}

void SessionContextManager::setResourcePressure(
    const std::string& session_id, const std::string& pressure) {
    std::lock_guard<std::mutex> lock(mutex_);
    getOrCreate(session_id).resource_pressure = pressure;
}

// =========================================================
// Context memory (replaces ContextManager statics)
// =========================================================

void SessionContextManager::rememberContextObject(
    const std::string& session_id,
    const UnifiedMemoryObject& obj) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);

    // Deduplicate by type + raw value
    for (auto& existing : s.recent_context) {
        if (existing.type == obj.type && existing.raw == obj.raw) {
            existing = obj;
            return;
        }
    }
    s.recent_context.push_back(obj);
}

std::optional<UnifiedMemoryObject> SessionContextManager::recallContextByType(
    const std::string& session_id,
    const std::string& type_tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* s = get(session_id);
    if (!s || s->recent_context.empty()) return std::nullopt;

    const UnifiedMemoryObject* best = nullptr;
    float best_score = 0.0f;

    for (const auto& obj : s->recent_context) {
        if (GRIM::toString(obj.type, GRIM::TypeNames) != type_tag)
            continue;
        float age = static_cast<float>(
            std::difftime(std::time(nullptr), static_cast<std::time_t>(obj.timestamp)));
        float freshness = std::max(0.0f, 1.0f - age / 180.0f);
        float score = obj.confidence * freshness;
        if (score > best_score) {
            best_score = score;
            best = &obj;
        }
    }
    return best ? std::optional<UnifiedMemoryObject>(*best) : std::nullopt;
}

void SessionContextManager::decayOldContext(
    const std::string& session_id, int seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.recent_context.erase(
        std::remove_if(s.recent_context.begin(), s.recent_context.end(),
            [&](const UnifiedMemoryObject& m) {
                auto age = std::difftime(
                    std::time(nullptr), static_cast<std::time_t>(m.timestamp));
                return age > seconds;
            }),
        s.recent_context.end());
}

// =========================================================
// Conversation history
// =========================================================

void SessionContextManager::addMessage(
    const std::string& session_id,
    const std::string& role,
    const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.conversation_history.push_back({role, content});
}

void SessionContextManager::setSystemPrompt(
    const std::string& session_id,
    const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    if (s.system_prompt_set) return;
    s.conversation_history.insert(
        s.conversation_history.begin(), {"system", prompt});
    s.system_prompt_set = true;
}

std::vector<SessionContextManager::ChatMessage>
SessionContextManager::getMessages(
    const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* s = get(session_id);
    if (!s) return {};
    return s->conversation_history;
}

void SessionContextManager::clearHistory(
    const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    s.conversation_history.clear();
    s.system_prompt_set = false;
}

void SessionContextManager::trimHistory(
    const std::string& session_id, size_t max_messages) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    size_t sys_count = s.system_prompt_set ? 1 : 0;
    if (s.conversation_history.size() > (max_messages + sys_count)) {
        size_t to_erase = s.conversation_history.size()
                          - (max_messages + sys_count);
        s.conversation_history.erase(
            s.conversation_history.begin() + static_cast<long>(sys_count),
            s.conversation_history.begin() + static_cast<long>(sys_count + to_erase));
    }
}

ConsoleHistory& SessionContextManager::displayHistory(
    const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& s = getOrCreate(session_id);
    if (!s.display_history) {
        throw std::runtime_error(
            "SessionContextManager display history is NULL for session: " + session_id);
    }
    return *s.display_history;
}

// =========================================================
// Legacy V1 snapshot projection
// =========================================================

GRIM::ContextSnapshot SessionContextManager::legacySnapshot(
    const std::string& session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    GRIM::ContextSnapshot snap;

    const auto* s = get(session_id);
    if (!s) return snap;

    snap.currentMood = s->mood;
    snap.conversationDepth = static_cast<int>(s->recent_context.size());
    snap.lastNlpCategory = s->last_nlp_category;
    snap.consecutiveCommands = s->consecutive_commands;

    int command_count = 0;
    std::time_t most_recent_cmd_time = 0;

    for (const auto& obj : s->recent_context) {
        if (obj.comm_type == GRIM::CommType::COMMAND) {
            if (snap.recentCommands.size() < 5) {
                snap.recentCommands.push_back(obj.raw);
            }
            command_count++;
            if (obj.timestamp > most_recent_cmd_time) {
                most_recent_cmd_time = obj.timestamp;
            }
            for (const auto& tag : obj.tags) {
                if (tag.find("nlp:") == 0) {
                    snap.lastNlpCategory = tag.substr(4);
                }
            }
        }
    }

    // Calculate consecutive commands from tail
    snap.consecutiveCommands = 0;
    for (auto it = s->recent_context.rbegin(); it != s->recent_context.rend(); ++it) {
        if (it->comm_type == GRIM::CommType::COMMAND) {
            snap.consecutiveCommands++;
        } else {
            break;
        }
    }
    snap.lastCommandTime = most_recent_cmd_time;

    return snap;
}

} // namespace GRIM::MMO
