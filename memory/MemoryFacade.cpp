#include "MemoryFacade.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../logger.hpp"

#include <algorithm>

static const std::string kDefaultSession = "default";

namespace GRIM::MMO {

// =========================================================
// Construction
// =========================================================

MemoryFacade::MemoryFacade(UnifiedMemoryStorage& storage)
    : storage_(storage) {}

// =========================================================
// Retrieval
// =========================================================

MemoryRetrievalResult MemoryFacade::retrieveForPrompt(
    const std::string& prompt,
    int max_memories) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryRetrievalResult result;
    result.context = SessionContextManager::instance().snapshot(kDefaultSession);

    // 1. Text search against storage
    {
        auto hits = storage_.search(prompt, max_memories);
        result.breadcrumbs.push_back({prompt, static_cast<int>(hits.size())});
        for (auto& m : hits) {
            const float score = m.confidence;
            result.hits.push_back({std::move(m), score});
        }
    }

    // 2. Tag search from recent NLP category if available
    if (!result.context.lastNlpCategory.empty()) {
        auto tag_hits = storage_.getByTag(result.context.lastNlpCategory);
        result.breadcrumbs.push_back(
            {"tag:" + result.context.lastNlpCategory,
             static_cast<int>(tag_hits.size())});
        for (auto& m : tag_hits) {
            // Deduplicate by ID
            bool dup = false;
            for (const auto& existing : result.hits) {
                if (existing.memory.id == m.id) { dup = true; break; }
            }
            if (!dup) {
                const float score = m.confidence;
                result.hits.push_back({std::move(m), score});
            }
        }
    }

    // 3. Truncate to max_memories (keep highest-confidence first)
    if (static_cast<int>(result.hits.size()) > max_memories) {
        std::partial_sort(
            result.hits.begin(),
            result.hits.begin() + max_memories,
            result.hits.end(),
            [](const MemoryRetrievalHit& a, const MemoryRetrievalHit& b) {
                return a.score > b.score;
            });
        result.hits.resize(max_memories);
    }

    return result;
}

std::vector<UnifiedMemoryObject> MemoryFacade::search(
    const std::string& query,
    int max_results) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.search(query, max_results);
}

std::vector<UnifiedMemoryObject> MemoryFacade::getByTag(
    const std::string& tag) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.getByTag(tag);
}

std::vector<UnifiedMemoryObject> MemoryFacade::getByTags(
    const std::vector<std::string>& tags,
    bool match_all) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.getByTags(tags, match_all);
}

// =========================================================
// Recording
// =========================================================

void MemoryFacade::recordInteraction(const UnifiedMemoryObject& obj)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Route through MemoryRouter to determine storage target
    auto decision = MemoryRouter::evaluate(obj);

    switch (decision.target) {
        case MemoryRouteTarget::LongTerm:
            storage_.storeLongTerm(obj);
            break;
        case MemoryRouteTarget::ShortTerm:
            storage_.storeShortTerm(obj);
            break;
        case MemoryRouteTarget::CommandExec:
            // Command-type memories go to short-term for context
            storage_.storeShortTerm(obj);
            break;
        case MemoryRouteTarget::Ignore:
            // Low-confidence or irrelevant — skip storage
            break;
    }

    // Also record in session context for immediate recall
    SessionContextManager::instance().rememberContextObject(kDefaultSession, obj);
}

void MemoryFacade::storeLongTerm(const UnifiedMemoryObject& obj)
{
    std::lock_guard<std::mutex> lock(mutex_);
    storage_.storeLongTerm(obj);
}

void MemoryFacade::storeShortTerm(const UnifiedMemoryObject& obj)
{
    std::lock_guard<std::mutex> lock(mutex_);
    storage_.storeShortTerm(obj);
}

// =========================================================
// Session context passthrough
// =========================================================

ContextSnapshot MemoryFacade::getContextSnapshot() const
{
    return SessionContextManager::instance().legacySnapshot(kDefaultSession);
}

void MemoryFacade::rememberContext(const UnifiedMemoryObject& obj)
{
    SessionContextManager::instance().rememberContextObject(kDefaultSession, obj);
}

void MemoryFacade::decayContext(int seconds)
{
    SessionContextManager::instance().decayOldContext(kDefaultSession, seconds);
}

// =========================================================
// Maintenance
// =========================================================

void MemoryFacade::flush()
{
    std::lock_guard<std::mutex> lock(mutex_);
    storage_.flush();
}

void MemoryFacade::compact()
{
    std::lock_guard<std::mutex> lock(mutex_);
    storage_.compactStorage();
}

} // namespace GRIM::MMO
