// MemoryBufferRotation — rotating 3-buffer memory pipeline.
//
// Three buffers rotate through the pipeline:
//   Working       – active context for the current request/session
//   Preprocessing – incoming data staged before merge
//   SyncAndClear  – syncs accumulated data to long-term, then clears
//
// Rotation order:
//   preprocess(obj) → Preprocessing buffer
//   mergeToWorking() → Preprocessing drains into Working
//   syncToLongTerm(storage) → Working snapshots into SyncAndClear,
//                              SyncAndClear persists to long-term, clears
//
// Thread-safe: all operations under mutex.
//======================================================//
#pragma once

#include "unified_memory.hpp"
#include <mutex>
#include <vector>

namespace GRIM {

class UnifiedMemoryStorage;  // forward

class MemoryBufferRotation {
public:
    static MemoryBufferRotation& instance();

    // Stage an incoming memory object for preprocessing.
    void preprocess(const UnifiedMemoryObject& obj);

    // Merge all preprocessed items into the working buffer.
    void mergeToWorking();

    // Snapshot working buffer → sync buffer, persist sync buffer to
    // long-term storage, then clear the sync buffer.
    void syncToLongTerm(UnifiedMemoryStorage& storage);

    // Read current working context for building precomposed metadata.
    std::vector<UnifiedMemoryObject> workingSnapshot() const;

    // Clear all buffers (shutdown / reset).
    void clear();

    struct Stats {
        size_t working_count   = 0;
        size_t preproc_count   = 0;
        size_t sync_count      = 0;
    };
    Stats stats() const;

private:
    MemoryBufferRotation() = default;

    mutable std::mutex mutex_;
    std::vector<UnifiedMemoryObject> working_;
    std::vector<UnifiedMemoryObject> preprocessing_;
    std::vector<UnifiedMemoryObject> sync_and_clear_;
};

} // namespace GRIM
