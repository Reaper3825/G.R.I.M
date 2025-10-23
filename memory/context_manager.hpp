#pragma once
#include "memory/memory_storage.hpp"
#include <unordered_map>
#include <optional>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace GRIM {

// ====================================================
// ContextManager — unified short-term entity + intent context
// ====================================================
class ContextManager {
public:
    // --- Existing utilities ---
    static void recordUsage(const std::string& category);
    static int usageCount(const std::string& category);
    static void setMemoryStorage(MemoryStorage* storage);
    static std::string getCurrentMood();

    // ====================================================
    // New: Contextual Memory Integration
    // ====================================================
    static void rememberContextObject(const MemoryObject& obj);
    static std::optional<MemoryObject> recallContextByType(const std::string& typeTag);
    static std::optional<MemoryObject> recallContextByIntent(const std::string& intentTag);
    static void decayOldContext(int seconds = 180);
    static void clearContext();

    // ====================================================
    // New: Pending Intent for multi-turn command flow
    // ====================================================
    struct PendingIntent {
        std::string command;      // e.g., "open_app"
        std::string missingTag;   // e.g., "TypeTag:App"
        std::chrono::steady_clock::time_point timestamp;
    };

    static void setPendingIntent(const PendingIntent& intent);
    static std::optional<PendingIntent> getPendingIntent();
    static void clearPendingIntent();

    // ====================================================
    // Accessors
    // ====================================================
    static std::vector<MemoryObject> getRecentContext();
    static void attachFeedbackMetadata(const std::string& command);

private:
    static std::mutex contextMutex;
    static std::mutex intentMutex;
};

} // namespace GRIM
