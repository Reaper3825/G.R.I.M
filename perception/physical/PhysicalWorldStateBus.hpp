#pragma once

#include "PhysicalWorldStateResult.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalWorldStateBus — single-producer (PhysicalWorldStateLoop),
//  multi-consumer latest-snapshot slot. Mirrors PhysicalSpatialGroundingBus
//  exactly so consumers see the same lifecycle.
//
//  Rule 20: Publish enforces invariants (source_frame_counter != 0); Pull
//  never silently fabricates a snapshot when none exists.
// ─────────────────────────────────────────────────────────────────────────────
class PhysicalWorldStateBus {
public:
    struct SnapshotView {
        PhysicalWorldStateSnapshot           snapshot;
        std::chrono::steady_clock::time_point published_at{};
    };

    static PhysicalWorldStateBus& Instance();

    // Producer side. Throws on source_frame_counter == 0 (counters start at
    // 1; zero indicates uninitialised data).
    void PublishPhysicalWorldStateSnapshotToBus(const PhysicalWorldStateSnapshot& snapshot);

    // Consumer side. Returns true iff a snapshot has ever been published AND
    // its source_frame_counter is newer than `last_seen_frame_counter`.
    bool PullLatestPhysicalWorldStateSnapshotView(SnapshotView& out,
                                                  uint64_t& last_seen_frame_counter) const;

    bool HasEverPublishedWorldStateSnapshot() const;

    void ResetPhysicalWorldStateBus();

private:
    PhysicalWorldStateBus() = default;
    PhysicalWorldStateBus(const PhysicalWorldStateBus&)            = delete;
    PhysicalWorldStateBus& operator=(const PhysicalWorldStateBus&) = delete;

    mutable std::mutex                    mutex_;
    PhysicalWorldStateSnapshot            latest_snapshot_{};
    std::chrono::steady_clock::time_point latest_time_{};
    bool                                  ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
