#include "memory/context_manager.hpp"
#include <mutex>
static std::mutex usageMutex;
namespace GRIM {

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
static MemoryStorage* g_memoryStoragePtr = nullptr;
void ContextManager::setMemoryStorage(MemoryStorage* storage) {
    g_memoryStoragePtr = storage;
}
std::string ContextManager::getCurrentMood() {
    // If you already track personality or affect elsewhere, hook it here.
    // For now, use a simple heuristic from usage or a default.
    if (usageMap.empty())
        return "Neutral";

    // Example: heavier command use implies more "Focused" mood
    int totalUsage = 0;
    for (const auto& [_, count] : usageMap)
        totalUsage += count;

    if (totalUsage < 10) return "Curious";
    if (totalUsage < 50) return "Focused";
    return "Tired";
}

} // namespace GRIM
