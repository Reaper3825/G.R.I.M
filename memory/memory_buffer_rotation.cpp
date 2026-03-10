// MemoryBufferRotation.cpp — rotating 3-buffer memory pipeline.
//======================================================//

#include "memory_buffer_rotation.hpp"
#include "unified_memory.hpp"
#include "logger.hpp"

namespace GRIM {

// ─── Singleton ────────────────────────────────────────────

MemoryBufferRotation& MemoryBufferRotation::instance() {
    static MemoryBufferRotation inst;
    return inst;
}

// ─── Preprocess ───────────────────────────────────────────

void MemoryBufferRotation::preprocess(const UnifiedMemoryObject& obj) {
    std::lock_guard<std::mutex> lock(mutex_);
    preprocessing_.push_back(obj);
}

// ─── Merge to Working ─────────────────────────────────────

void MemoryBufferRotation::mergeToWorking() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (preprocessing_.empty()) return;

    working_.insert(working_.end(),
                    std::make_move_iterator(preprocessing_.begin()),
                    std::make_move_iterator(preprocessing_.end()));
    preprocessing_.clear();

    LOG_DEBUG("MemoryRotation",
              "Merged preprocessing → working (working now has "
              + std::to_string(working_.size()) + " items)");
}

// ─── Sync to Long-Term ───────────────────────────────────

void MemoryBufferRotation::syncToLongTerm(UnifiedMemoryStorage& storage) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Move working → sync buffer
    sync_and_clear_.insert(sync_and_clear_.end(),
                           std::make_move_iterator(working_.begin()),
                           std::make_move_iterator(working_.end()));
    working_.clear();

    // Persist sync buffer to long-term storage
    for (const auto& obj : sync_and_clear_) {
        storage.storeLongTerm(obj);
    }

    size_t synced = sync_and_clear_.size();
    sync_and_clear_.clear();

    if (synced > 0) {
        LOG_DEBUG("MemoryRotation",
                  "Synced " + std::to_string(synced)
                  + " items to long-term storage");
    }
}

// ─── Working Snapshot ─────────────────────────────────────

std::vector<UnifiedMemoryObject> MemoryBufferRotation::workingSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return working_;
}

// ─── Clear ────────────────────────────────────────────────

void MemoryBufferRotation::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    working_.clear();
    preprocessing_.clear();
    sync_and_clear_.clear();
}

// ─── Stats ────────────────────────────────────────────────

MemoryBufferRotation::Stats MemoryBufferRotation::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {working_.size(), preprocessing_.size(), sync_and_clear_.size()};
}

} // namespace GRIM
