#pragma once

#include "PhysicalMonocularDepthEstimator.hpp"
#include "PhysicalSpatialGrounder.hpp"

#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalSpatialGroundingLoop — Stage-3 mainloop integration point.
//
//  The mainloop calls TickPhysicalSpatialGrounding() exactly once per frame
//  immediately AFTER TickPhysicalPerceptionPrimitives(). Tick only schedules
//  a bounded latest-result worker and never waits for inference. On the first
//  worker invocation the subsystem lazy-initialises:
//    - constructs the depth estimator + spatial grounder
//    - applies any configuration registered via RequestConfigure*() before
//      the first Tick (otherwise: depth estimator stays NoModelConfigured;
//      grounder is auto-configured with default parameters)
//
//  Every subsequent worker invocation:
//    - PullLatestPhysicalPerceptionResults from PhysicalPerceptionPrimitiveBus
//    - Use the exact source-frame view pinned into that Stage-2 result
//    - If a new coherent result is available:
//        * RouteFrameToPhysicalMonocularDepthEstimator
//        * RouteDepthAndTracksToPhysicalSpatialGrounder
//        * PublishPhysicalSpatialGroundingResultsToBus
//
//  No other file should touch the depth estimator or grounder during the
//  main loop. UI / config mutation goes through Request* below; thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalSpatialGroundingEnableFlags {
    bool depth_estimator = true;
    bool spatial_grounder = true;
};

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalSpatialGrounding();

// Stop processing and tear down the operator instances.
void ShutdownPhysicalSpatialGrounding();

bool IsPhysicalSpatialGroundingRunning();

// ── Read-only state ─────────────────────────────────────────────────────────

PhysicalSpatialGroundingEnableFlags GetPhysicalSpatialGroundingEnableFlags();
std::string                          GetLastPhysicalSpatialGroundingError();
uint64_t                             GetPhysicalSpatialGroundingTickCount();
uint64_t                             GetPhysicalSpatialGroundingProcessedCount();

// ── UI-facing requests (thread-safe) ────────────────────────────────────────

void RequestSetPhysicalSpatialGroundingEnableFlags(
    const PhysicalSpatialGroundingEnableFlags& flags);

// Atomic config swap. Each Request* re-loads the model (or resets to
// NoModelConfigured if path is empty). Throws on bad config — Rule 20.
void RequestConfigurePhysicalMonocularDepthEstimator(
    const PhysicalMonocularDepthEstimatorConfig& cfg);

void RequestConfigurePhysicalSpatialGrounder(
    const PhysicalSpatialGrounderConfig& cfg);

}}} // namespace GRIM::Perception::Physical
