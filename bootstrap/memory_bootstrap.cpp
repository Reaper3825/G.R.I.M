#include "bootstrap.hpp"

#include "../logger.hpp"
#include "../memory/unified_memory.hpp"
#include "../resources.hpp"

#include <filesystem>
#include <stdexcept>

void bootstrapMemorySubsystem(GRIM::UnifiedMemoryStorage& memoryStorage)
{
    memoryStorage.initialize(
        (std::filesystem::path(getGrimRootDir()) / "data" / "memories.fb").string());
    LOG_PHASE("Memory system initialized", true);

    if (!g_orchestrator) {
        throw std::runtime_error(
            "bootstrapMemorySubsystem: MMO orchestrator is NULL after core bootstrap");
    }
    if (g_memoryFacade) {
        throw std::runtime_error(
            "bootstrapMemorySubsystem: MemoryFacade is already initialized");
    }

    g_memoryFacade = new GRIM::MMO::MemoryFacade(memoryStorage);
    g_orchestrator->setMemoryFacade(g_memoryFacade);
    LOG_PHASE("MemoryFacade wired to orchestrator", true);
}