#include "memory/context_manager.hpp"
#include "ai/personality_manager.hpp"
#include "logger.hpp"
#include <algorithm>

static std::mutex usageMutex;

namespace GRIM {

// ====================================================
// Usage tracking
// ====================================================
static std::unordered_map<std::string, int> usageMap;

void ContextManager::recordUsage(const std::string& category) {
    std::lock_guard<std::mutex> lock(usageMutex);
    usageMap[category]++;
}

int ContextManager::usageCount(const std::string& category) {
    std::lock_guard<std::mutex> lock(usageMutex);
    auto it = usageMap.find(category);
    return (it != usageMap.end()) ? it->second : 0;
}

// ====================================================
// Memory + mood bridge
// ====================================================
static MemoryStorage* g_memoryStoragePtr = nullptr;

void ContextManager::setMemoryStorage(MemoryStorage* storage) {
    g_memoryStoragePtr = storage;
}

std::string ContextManager::getCurrentMood() {
    auto& state = PersonalityManager::get();
    return PersonalityManager::moodToString(state.mood);
}

// ====================================================
// Contextual Memory Layer
// ====================================================
std::mutex ContextManager::contextMutex;
std::mutex ContextManager::intentMutex;

namespace {
    std::vector<MemoryObject> recentContext;
    std::optional<ContextManager::PendingIntent> pendingIntent;
}

void ContextManager::rememberContextObject(const MemoryObject& obj) {
    std::lock_guard<std::mutex> lock(contextMutex);

    // Avoid duplicates by TypeTag + value
    auto it = std::find_if(recentContext.begin(), recentContext.end(), [&](const MemoryObject& m) {
    return m.type == obj.type && m.raw == obj.raw;
    });


    if (it != recentContext.end()) {
        *it = obj;
    } else {
        recentContext.push_back(obj);
    }

    LOG_TRACE("Context", "Stored context object: " + obj.raw + 
          " [" + GRIM::toString(obj.type, GRIM::TypeNames) + "]");
}

std::optional<MemoryObject> ContextManager::recallContextByType(const std::string& typeTag) {
    std::lock_guard<std::mutex> lock(contextMutex);
    if (recentContext.empty()) return std::nullopt;

    auto now = std::chrono::steady_clock::now();
    MemoryObject* best = nullptr;
    float bestScore = 0.0f;

    for (auto& obj : recentContext) {
        // Compare using enum-to-string helper
        if (GRIM::toString(obj.type, GRIM::TypeNames) != typeTag)
            continue;

        // Calculate age using time_t difference
        float age = static_cast<float>(std::difftime(std::time(nullptr), obj.timestamp));
        float freshness = std::max(0.0f, 1.0f - age / 180.0f);
        float score = obj.confidence * freshness;

        if (score > bestScore) {
            bestScore = score;
            best = &obj;
        }
    }

    return best ? std::optional<MemoryObject>(*best) : std::nullopt;
}


std::optional<MemoryObject> ContextManager::recallContextByIntent(const std::string& intentTag) {
    std::lock_guard<std::mutex> lock(contextMutex);
    if (recentContext.empty()) return std::nullopt;

    for (auto it = recentContext.rbegin(); it != recentContext.rend(); ++it) {
        if (GRIM::toString(it->intent, GRIM::IntentNames) == intentTag)

            return *it;
    }
    return std::nullopt;
}

void ContextManager::decayOldContext(int seconds) {
    std::lock_guard<std::mutex> lock(contextMutex);
    auto now = std::chrono::steady_clock::now();
    recentContext.erase(std::remove_if(recentContext.begin(), recentContext.end(),
        [&](const MemoryObject& m) {
            auto age = std::difftime(std::time(nullptr), m.timestamp);
            return age > seconds;
        }), recentContext.end());
}

void ContextManager::clearContext() {
    std::lock_guard<std::mutex> lock(contextMutex);
    recentContext.clear();
}

std::vector<MemoryObject> ContextManager::getRecentContext() {
    std::lock_guard<std::mutex> lock(contextMutex);
    return recentContext;
}

// ====================================================
// Pending Intent System
// ====================================================
void ContextManager::setPendingIntent(const PendingIntent& intent) {
    std::lock_guard<std::mutex> lock(intentMutex);
    pendingIntent = intent;
    LOG_DEBUG("Context", "Set pending intent for: " + intent.command);
}

std::optional<ContextManager::PendingIntent> ContextManager::getPendingIntent() {
    std::lock_guard<std::mutex> lock(intentMutex);
    return pendingIntent;
}

void ContextManager::clearPendingIntent() {
    std::lock_guard<std::mutex> lock(intentMutex);
    if (pendingIntent.has_value())
        LOG_TRACE("Context", "Cleared pending intent: " + pendingIntent->command);
    pendingIntent.reset();
}

// ====================================================
// Feedback Integration Helper
// ====================================================
void ContextManager::attachFeedbackMetadata(const std::string& command) {
    std::lock_guard<std::mutex> lock(contextMutex);
    if (g_memoryStoragePtr) {
        try {
            MemoryObject feedback;
            feedback.id = MemoryObject::generateUUID();
            feedback.timestamp = std::time(nullptr);
            feedback.source = GRIM::SourceTag::GrimInternal;
            feedback.type = GRIM::TypeTag::Command;
            feedback.intent = GRIM::IntentTag::Inform;
            feedback.context = GRIM::ContextTag::Conversation;
            feedback.raw = command;
            feedback.normalized = command;
            feedback.confidence = 1.0f;
            feedback.tags = {"context_feedback"};

            g_memoryStoragePtr->storeShortTerm(feedback);
            LOG_DEBUG("Context", "Attached feedback metadata for command: " + command);
        } catch (const std::exception& e) {
            LOG_ERROR("Context", std::string("Failed to attach feedback metadata: ") + e.what());
        }
    }
}

} // namespace GRIM
