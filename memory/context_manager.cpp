#include "memory/context_manager.hpp"
#include "ai/personality_manager.hpp"
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
    // ? FIX: Use actual PersonalityManager state instead of heuristics
    auto& state = PersonalityManager::get();
    return PersonalityManager::moodToString(state.mood);
}

} // namespace GRIM
