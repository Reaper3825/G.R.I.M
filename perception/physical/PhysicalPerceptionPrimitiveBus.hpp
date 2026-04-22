#pragma once

#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalPerceptionPrimitiveBus
//
//  Single-producer, multi-consumer latest-result slot. Mirrors the
//  PhysicalFrameBus design exactly so consumers (UI today; model context
//  matrix in later stages) see the same lifecycle.
//
//  Producer: PhysicalPerceptionPrimitivesLoop, exactly once per Tick.
//  Consumers: any number of read-only pullers; counter-based version check
//  prevents redundant work when nothing has changed.
//
//  Rule 20: Publish enforces invariants; pull never silently fabricates a
//  result when none exists.
// ─────────────────────────────────────────────────────────────────────────────
class PhysicalPerceptionPrimitiveBus {
public:
    struct ResultsView {
        PhysicalPerceptionPrimitiveResults    results;
        std::chrono::steady_clock::time_point published_at{};
    };

    static PhysicalPerceptionPrimitiveBus& Instance();

    // Producer side. Throws if source_frame_counter == 0 (counters start at 1
    // in PhysicalFrameBus, so 0 indicates uninitialised data).
    void PublishPhysicalPerceptionResultsToBus(const PhysicalPerceptionPrimitiveResults& results);

    // Consumer side. Returns true iff a result has ever been published AND
    // its source_frame_counter is newer than `last_seen_frame_counter`.
    // Updates `last_seen_frame_counter` on hit.
    bool PullLatestPhysicalPerceptionResultsView(ResultsView& out,
                                                 uint64_t& last_seen_frame_counter) const;

    bool HasEverPublishedResults() const;

    // Drop all state (e.g. on subsystem shutdown or active source change).
    void ResetPhysicalPerceptionPrimitiveBus();

private:
    PhysicalPerceptionPrimitiveBus() = default;
    PhysicalPerceptionPrimitiveBus(const PhysicalPerceptionPrimitiveBus&)            = delete;
    PhysicalPerceptionPrimitiveBus& operator=(const PhysicalPerceptionPrimitiveBus&) = delete;

    mutable std::mutex                          mutex_;
    PhysicalPerceptionPrimitiveResults          latest_results_{};
    std::chrono::steady_clock::time_point       latest_time_{};
    bool                                        ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
