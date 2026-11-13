#pragma once

#include "PhysicalHandGestureResult.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace GRIM { namespace Perception { namespace Physical {

class PhysicalHandGestureBus {
public:
    struct SnapshotView {
        PhysicalHandGestureSnapshot snapshot;
        std::chrono::steady_clock::time_point published_at{};
    };

    static PhysicalHandGestureBus& Instance();

    void PublishPhysicalHandGestureSnapshot(
        const PhysicalHandGestureSnapshot& snapshot);
    bool PullLatestPhysicalHandGestureSnapshot(
        SnapshotView& out, uint64_t& last_seen_publish_sequence) const;
    bool HasEverPublishedSnapshot() const;
    void ResetPhysicalHandGestureBus();

private:
    PhysicalHandGestureBus() = default;
    PhysicalHandGestureBus(const PhysicalHandGestureBus&) = delete;
    PhysicalHandGestureBus& operator=(const PhysicalHandGestureBus&) = delete;

    mutable std::mutex mutex_;
    PhysicalHandGestureSnapshot latest_{};
    std::chrono::steady_clock::time_point latest_time_{};
    uint64_t next_publish_sequence_ = 1;
    bool ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
