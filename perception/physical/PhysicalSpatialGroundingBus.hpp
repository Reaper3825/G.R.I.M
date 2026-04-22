#pragma once

#include "PhysicalSpatialGroundingResult.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalSpatialGroundingBus
//
//  Single-producer, multi-consumer latest-result slot. Mirrors
//  PhysicalPerceptionPrimitiveBus exactly so the UI / model context matrix /
//  any downstream consumer sees the same lifecycle for Stage-3 output as
//  for Stage-2.
//
//  Producer: PhysicalSpatialGroundingLoop, exactly once per Tick.
//  Consumers: any number of read-only pullers; counter-based version check
//  prevents redundant work when nothing has changed.
//
//  Rule 20: Publish enforces invariants; pull never silently fabricates a
//  result when none exists.
// ─────────────────────────────────────────────────────────────────────────────
class PhysicalSpatialGroundingBus {
public:
    struct ResultsView {
        PhysicalSpatialGroundingResults       results;
        std::chrono::steady_clock::time_point published_at{};
    };

    static PhysicalSpatialGroundingBus& Instance();

    // Producer side. Throws if source_frame_counter == 0 (FrameBus counters
    // start at 1, so 0 indicates uninitialised data).
    void PublishPhysicalSpatialGroundingResultsToBus(
        const PhysicalSpatialGroundingResults& results);

    // Consumer side. Returns true iff a result has ever been published AND
    // its source_frame_counter is newer than `last_seen_frame_counter`.
    // Updates `last_seen_frame_counter` on hit.
    bool PullLatestPhysicalSpatialGroundingResultsView(
        ResultsView& out, uint64_t& last_seen_frame_counter) const;

    bool HasEverPublishedSpatialGroundingResults() const;

    // Drop all state (e.g. on subsystem shutdown).
    void ResetPhysicalSpatialGroundingBus();

private:
    PhysicalSpatialGroundingBus() = default;
    PhysicalSpatialGroundingBus(const PhysicalSpatialGroundingBus&)            = delete;
    PhysicalSpatialGroundingBus& operator=(const PhysicalSpatialGroundingBus&) = delete;

    mutable std::mutex                          mutex_;
    PhysicalSpatialGroundingResults             latest_results_{};
    std::chrono::steady_clock::time_point       latest_time_{};
    bool                                        ever_published_ = false;
};

}}} // namespace GRIM::Perception::Physical
