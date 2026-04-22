#pragma once

#include "PhysicalImageClassifier.hpp"
#include "PhysicalInstanceSegmenter.hpp"
#include "PhysicalObjectDetector.hpp"
#include "PhysicalPoseKeypointEstimator.hpp"
#include "PhysicalSceneTextReader.hpp"
#include "PhysicalSemanticSegmenter.hpp"
#include "PhysicalFacialExpressionDetector.hpp"
#include "PhysicalEntityTracker.hpp"
#include "PhysicalClassPolicy.hpp"

#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalPerceptionPrimitivesLoop — Stage-2 integration point.
//
//  The mainloop calls TickPhysicalPerceptionPrimitives() exactly once per
//  frame (immediately AFTER TickPhysicalEnvironment()). On the first call
//  the subsystem lazy-initialises:
//    - constructs the five operator instances
//    - applies the configuration registered via
//      ConfigurePhysicalPerceptionPrimitives*() OR the empty defaults
//      (which leave every operator in NoModelConfigured)
//
//  Every subsequent call:
//    - PullLatestFrameView from PhysicalFrameBus; if the counter has not
//      advanced, do nothing (no redundant inference)
//    - For each operator whose state == ModelLoaded AND whose enabled flag
//      is true, RouteFrameTo<Operator>() and collect the result
//    - PublishPhysicalPerceptionResultsToBus() once per Tick with the
//      coherent set of results
//
//  No other file should touch the operators directly during the main loop.
//  All UI / config mutation goes through the Request* functions below; they
//  are thread-safe.
// ─────────────────────────────────────────────────────────────────────────────

// One enable flag per operator. Defaults: ALL true. An operator with
// enabled=true but state=NoModelConfigured stays idle; the UI surfaces
// the state explicitly so the user knows why no result is showing.
struct PhysicalPerceptionPrimitivesEnableFlags {
    bool object_detector    = true;
    bool semantic_segmenter = true;
    bool image_classifier   = true;
    bool pose_estimator     = true;
    bool scene_text_reader  = true;
    bool facial_expression_detector = true;
    // Entity tracker consumes the object detector's output; if
    // object_detector is off OR has no model loaded, the tracker still runs
    // but sees an empty detection list (it will simply age out tracks).
    bool entity_tracker     = true;
    // Instance segmenter (SAM 2) consumes the object detector's boxes as
    // prompts. Same gating as the tracker: with no detections this frame the
    // operator skips the encoder pass entirely (cheap no-op).
    bool instance_segmenter = true;
    // Class policy applies the merge map + priority cutoff to the per-
    // operator results in place. Disabling it leaves the per-operator
    // labels untouched and emits an empty ranked summary.
    bool class_policy       = true;
};

// Mainloop entry point. Cheap; safe to call every frame. Lazy-inits.
void TickPhysicalPerceptionPrimitives();

// Stop processing and tear down the operator instances.
void ShutdownPhysicalPerceptionPrimitives();

bool IsPhysicalPerceptionPrimitivesRunning();

// ── Read-only state ─────────────────────────────────────────────────────────

PhysicalPerceptionPrimitivesEnableFlags GetPhysicalPerceptionPrimitivesEnableFlags();
std::string                              GetLastPhysicalPerceptionPrimitivesError();
uint64_t                                 GetPhysicalPerceptionPrimitivesTickCount();
uint64_t                                 GetPhysicalPerceptionPrimitivesProcessedCount();

// ── UI-facing requests (thread-safe) ────────────────────────────────────────

void RequestSetPhysicalPerceptionPrimitivesEnableFlags(
    const PhysicalPerceptionPrimitivesEnableFlags& flags);

// Atomic per-operator config swap. Each Request* call re-loads the model
// (or resets to NoModelConfigured if the path is empty). Throws on bad
// config — Rule 20.
void RequestConfigurePhysicalObjectDetector(const PhysicalObjectDetectorConfig& cfg);
void RequestConfigurePhysicalSemanticSegmenter(const PhysicalSemanticSegmenterConfig& cfg);
void RequestConfigurePhysicalImageClassifier(const PhysicalImageClassifierConfig& cfg);
void RequestConfigurePhysicalPoseKeypointEstimator(const PhysicalPoseKeypointEstimatorConfig& cfg);
void RequestConfigurePhysicalSceneTextReader(const PhysicalSceneTextReaderConfig& cfg);
void RequestConfigurePhysicalFacialExpressionDetector(const PhysicalFacialExpressionDetectorConfig& cfg);
void RequestConfigurePhysicalEntityTracker(const PhysicalEntityTrackerConfig& cfg);
void RequestConfigurePhysicalInstanceSegmenter(const PhysicalInstanceSegmenterConfig& cfg);
void RequestConfigurePhysicalClassPolicy(const PhysicalClassPolicyConfig& cfg);

}}} // namespace GRIM::Perception::Physical
