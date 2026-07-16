#pragma once

#include "PhysicalLocalizationResult.hpp"
#include "PhysicalOccupancyGridMapper.hpp"
#include "PhysicalVisualOdometer.hpp"

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalLocalizationLoop — Stage-5 mainloop integration point.
//
//  The mainloop calls TickPhysicalLocalization() exactly once per frame.
//  Tick only schedules a bounded latest-frame worker and never waits for
//  odometry. It is invoked immediately AFTER TickPhysicalSpatialGrounding() (and BEFORE
//  TickPhysicalWorldState() so a future revision of the world-state
//  builder can stamp every entity into the world frame).
//
//  On the first worker invocation:
//    * Constructs PhysicalVisualOdometer + PhysicalOccupancyGridMapper.
//    * Pulls camera intrinsics from PhysicalCameraCalibrator (if available)
//      and feeds them into the odometer. If no calibration is loaded,
//      the loop publishes a Failed snapshot every tick with a clear
//      reason — the UI will display it.
//
//  Every subsequent worker invocation:
//    * PullLatestFrameView from PhysicalFrameBus.
//    * If calibration just became available, push it into the odometer.
//    * If a new frame arrived, RouteFrameToPhysicalVisualOdometer.
//    * Update the occupancy grid from the new pose.
//    * Build + publish a PhysicalLocalizationSnapshot to
//      PhysicalLocalizationBus.
//
//  No other file touches the odometer or grid mapper during the main loop.
//  UI / config mutation goes through Request* below; thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalLocalization();

// Stop processing and tear down odometer + grid.
void ShutdownPhysicalLocalization();

bool IsPhysicalLocalizationRunning();

// ── Read-only state ─────────────────────────────────────────────────────────

std::string GetLastPhysicalLocalizationError();
uint64_t    GetPhysicalLocalizationTickCount();
uint64_t    GetPhysicalLocalizationProcessedCount();

PhysicalVisualOdometerConfig         GetPhysicalLocalizationOdometerConfigSnapshot();
PhysicalOccupancyGridMapperConfig    GetPhysicalLocalizationGridConfigSnapshot();

// ── UI-facing requests (thread-safe) ────────────────────────────────────────

// Atomic config swap. Throws on bad config — Rule 20.
void RequestConfigurePhysicalLocalizationOdometer(
    const PhysicalVisualOdometerConfig& cfg);

void RequestConfigurePhysicalLocalizationGrid(
    const PhysicalOccupancyGridMapperConfig& cfg);

// Drop the world frame and the occupancy map. Next valid frame becomes
// the new anchor. Useful for "I just teleported the camera" scenarios.
void RequestResetPhysicalLocalization();

}}} // namespace GRIM::Perception::Physical
