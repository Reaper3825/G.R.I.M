#pragma once
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <ctime>
#include <nlohmann/json.hpp>
#include "intent.hpp"
#include "memory_storage.hpp"
#include "logger.hpp"

namespace GRIM {

struct ContextFrame {
    std::string userInput;
    std::string aiReply;
    std::vector<std::string> recentTags;
    std::vector<std::string> activeIntents;
    float avgConfidence = 1.0f;
    std::time_t timestamp = std::time(nullptr);

    nlohmann::json toJSON() const {
        nlohmann::json j;
        j["user_input"] = userInput;
        j["ai_reply"] = aiReply;
        j["recent_tags"] = recentTags;
        j["active_intents"] = activeIntents;
        j["avg_confidence"] = avgConfidence;
        j["timestamp"] = static_cast<long long>(timestamp);
        return j;
    }
};

class ContextManager {
public:
    static void recordUsage(const std::string& category);
    static void setMemoryStorage(MemoryStorage* storage);
    static int usageCount(const std::string& category);
    static void recordInteraction(const std::string& userInput,
                                  const std::string& aiReply,
                                  const Intent& lastIntent);
    static ContextFrame buildFrame();
    static std::string buildFrameJSON();
    static void purgeOldFrames(size_t keepLast = 15);
    static void saveToDisk(const std::string& path);
    static void loadFromDisk(const std::string& path);
    // Returns current mood string for RL / personality feedback
    static std::string getCurrentMood();

private:
    static std::mutex mtx;
    static std::deque<ContextFrame> frames;
    static MemoryStorage* memory; // ✅ new
    static inline std::unordered_map<std::string, int> usageMap{};
};


} // namespace GRIM
