#pragma once

#include "PhysicalLocalizationResult.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalLocalizationBus — single-producer (PhysicalLocalizationLoop),
//  multi-consumer latest-snapshot slot. Mirrors PhysicalWorldStateBus.
//
//  Rule 20:
//    * Publish enforces invariants (source_frame_counter != 0 EXCEPT when
//      tracking_state == Uninitialized — the first lazy-init publish is
//      allowed to carry counter 0 so a UI can show "Uninitialized" before
//      a single frame has flowed through).
//    * Pull never silently fabricates a snapshot when none exists.
// ─────────────────────────────────────────────────────────────────────────────
class PhysicalLocalizationBus {
public:
    struct SnapshotView {
        PhysicalLocalizationSnapshot         snapshot;
        std::chrono::steady_clock::time_point published_at{};
    };

    static PhysicalLocalizationBus& Instance();

    // Producer side. Throws if the snapshot is in tracking_state==Tracking
    // but source_frame_counter == 0 — that combination is impossible and
    // would mislead consumers about provenance.
    void PublishPhysicalLocalizationSnapshotToBus(const PhysicalLocalizationSnapshot& snapshot);

    // Consumer side. Returns true iff a snapshot has ever been published
    // AND it is newer than the caller's seen counter (or the caller has
    // never seen one).
    //
    // We can't use source_frame_counter alone because Uninitialized
    // snapshots legitimately carry 0. So we also bump an internal
    // monotonic publish_sequence counter.
    bool PullLatestPhysicalLocalizationSnapshotView(SnapshotView& out,
                                                    uint64_t& last_seen_publish_sequence) const;

    bool HasEverPublishedLocalizationSnapshot() const;

    void ResetPhysicalLocalizationBus();

private:
    PhysicalLocalizationBus() = default;
    PhysicalLocalizationBus(const PhysicalLocalizationBus&)            = delete;
    PhysicalLocalizationBus& operator=(const PhysicalLocalizationBus&) = delete;

    mutable std::mutex                    mutex_;
    PhysicalLocalizationSnapshot          latest_snapshot_{};
    std::chrono::steady_clock::time_point latest_time_{};
    uint64_t                              publish_sequence_ = 0;
    bool                                  ever_published_   = false;
};

}}} // namespace GRIM::Perception::Physical
