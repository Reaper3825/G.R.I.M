#include "command_memory_services.hpp"
#include "unified_memory.hpp"
#include "memory_buffer_rotation.hpp"

extern GRIM::UnifiedMemoryStorage g_memoryStorage;
namespace GRIM::CommandMemory {
void stageManual(const std::string& raw) {
    UnifiedMemoryObject obj;
    obj.id = UnifiedMemoryObject::generateID();
    obj.timestamp = static_cast<uint64_t>(std::time(nullptr));
    obj.domain = MemoryDomain::USER;
    obj.type = TypeTag::STRING;
    obj.context = ContextType::CONVERSATION;
    obj.confidence = 0.98f;
    obj.raw = raw;
    obj.normalized = "remember " + raw;
    obj.tags = {"manual", "remember"};
    MemoryBufferRotation::instance().preprocess(obj);
}
std::vector<Hit> search(const std::string& query) {
    std::vector<Hit> hits;
    for (const auto& obj : g_memoryStorage.search(query)) hits.push_back({obj.id, obj.raw});
    return hits;
}
bool flushIfPresent(uint64_t id) {
    if (!g_memoryStorage.getById(id)) return false;
    g_memoryStorage.flush();
    return true;
}
}
