#pragma once

#include "PhysicalWorldStateBuilder.hpp"

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalWorldStateLoop — Stage-4 mainloop integration point.
//
//  The mainloop calls TickPhysicalWorldState() exactly once per frame
//  immediately AFTER TickPhysicalSpatialGrounding(). On the first call the
//  subsystem lazy-initialises with PhysicalWorldStateBuilderConfig defaults
//  (overridable via RequestConfigurePhysicalWorldStateBuilder before or
//  after the first Tick).
//
//  Every subsequent call:
//    - PullLatestPhysicalPerceptionResults from PhysicalPerceptionPrimitiveBus
//    - PullLatestPhysicalSpatialGroundingResults from PhysicalSpatialGroundingBus
//    - If both advanced AND share the same source_frame_counter:
//        * BuildPhysicalWorldStateSnapshot
//        * PublishPhysicalWorldStateSnapshotToBus
//
//  No other file touches the world-state builder during the main loop.
//  UI / config mutation goes through Request* below; thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalWorldState();

// Stop processing and tear down builder state.
void ShutdownPhysicalWorldState();

bool IsPhysicalWorldStateRunning();

// ── Read-only state ─────────────────────────────────────────────────────────

std::string GetLastPhysicalWorldStateError();
uint64_t    GetPhysicalWorldStateTickCount();
uint64_t    GetPhysicalWorldStateProcessedCount();

PhysicalWorldStateBuilderConfig GetPhysicalWorldStateBuilderConfigSnapshot();

// ── UI-facing requests (thread-safe) ────────────────────────────────────────

// Atomic config swap. Throws on bad config — Rule 20.
void RequestConfigurePhysicalWorldStateBuilder(
    const PhysicalWorldStateBuilderConfig& cfg);

}}} // namespace GRIM::Perception::Physical
