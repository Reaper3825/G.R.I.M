#pragma once

#include <cstdint>
#include <string>

namespace GRIMText::Training {

struct TrainingContext;

struct MemorySnapshot {
    int device = -1;
    std::string device_name;
    std::uint64_t total_bytes = 0;
    std::uint64_t free_bytes = 0;
};

MemorySnapshot captureMemorySnapshotOrThrow();
void MemorySnapshotReady(TrainingContext& ctx);

} // namespace GRIMText::Training

