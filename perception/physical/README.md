# Physical Perception File Taxonomy

> Namespace: `GRIM::Perception::Physical`  
> Architecture guide: [`docs/PHYSICAL_PERCEPTION_SUBSYSTEM.md`](../../docs/PHYSICAL_PERCEPTION_SUBSYSTEM.md)

This folder is the real-world camera perception stack. Keep the files grouped
by **pipeline ownership**, not by model family. A stage loop owns its operators,
publishes exactly one bus snapshot, and downstream stages only read buses.

## Ownership Rules

1. `Tick*Loop` files are orchestration boundaries. They may own operators and
   mutable stage state.
2. `*Bus` files are latest-snapshot handoff points. They do not run inference.
3. `*Result` / `*DepthMap` files are data contracts. They must not own runtime
   resources, model handles, worker threads, or global singletons.
4. Operator files own exactly one algorithm/model family and expose a loud
   `Load*`, `RouteFrameTo*`, `Reset*`, state, and last-error surface.
5. Cross-stage fusion belongs in the downstream stage, never in the upstream
   producer.
6. New files must land in one of the taxonomy rows below. If a file fits two
   rows, split it before adding it.

## Shared Contracts

| Files | Responsibility |
|---|---|
| `PhysicalSceneStability.hpp` | Frame-change signal used by cadence/cache logic. |
| `PhysicalImageOperatorState.hpp` | Shared model/operator state enum. |
| `PhysicalClassPolicy.*` | Cross-operator class merge/prioritization policy used by Stage 2. |

## Stage 1 — Environment / Frame Producer

| Files | Responsibility |
|---|---|
| `PhysicalEnvironmentLoop.*` | Mainloop entry point; drains the active camera stream, conditions frames, publishes `PhysicalFrameBus`. |
| `PhysicalCameraDirectory.*` | Enumerates selectable camera/NIC/hub sources. |
| `PhysicalCameraSource.*` | Describes one selectable source row. |
| `PhysicalCameraStream.*` | Worker-threaded `cv::VideoCapture` wrapper. |
| `PhysicalFrameConditioner.*` | Quality gate, stabilization/denoise/exposure/deblur/resize/color conversion, scene-stability timing. |
| `PhysicalFrameBus.*` | Single-producer/multi-consumer latest-frame packet bus. |
| `PhysicalNicScan.hpp`, `PhysicalNicScan_win32.cpp`, `PhysicalNicScan_macos.mm`, `PhysicalNicScan_linux.cpp` | Platform NIC enumeration. Only the host implementation is compiled. |
| `PhysicalEnvironmentLogTag.hpp` | Stage-1 log tag. |

### Frame bus packet contract

`PhysicalFrameBus` publishes an immutable `PhysicalFramePacket`. Each
`FrameView` pins that packet with `std::shared_ptr<const PhysicalFramePacket>`
and exposes shallow `cv::Mat` headers into the packet's `raw_image` and
`model_image` buffers.

Consumers MUST treat `FrameView::raw_image`, `FrameView::model_image`, and
`FrameView::image` as read-only. If a consumer needs in-place OpenCV mutation,
it must `clone()` into a local scratch image first.

## Stage 1 Auxiliary — Calibration

| Files | Responsibility |
|---|---|
| `PhysicalCameraCalibrator.*` | Pulls Stage-1 frames, detects calibration patterns, computes intrinsics/distortion. |
| `PhysicalCalibrationPattern.*` | Pattern geometry + lighting-robust detection helpers. |
| `PhysicalCalibrationStore.*` | Disk persistence for calibration results. |

Calibration is an auxiliary Stage-1 consumer and a Stage-5 prerequisite; it is
not part of the Stage-2 inference path.

## Stage 2 — Perception Primitives

| Files | Responsibility |
|---|---|
| `PhysicalPerceptionPrimitivesLoop.*` | Mainloop entry point; owns and schedules all Stage-2 operators. |
| `PhysicalPerceptionPrimitiveBus.*` | Latest coherent Stage-2 result bundle. |
| `PhysicalPerceptionPrimitiveResult.hpp` | Stage-2 result contract and telemetry. |
| `PhysicalPerceptionPrimitivesLogTag.hpp` | Stage-2 log tag. |

### Stage-2 Operators

| Files | Responsibility |
|---|---|
| `PhysicalObjectDetector.*` | Object boxes/classes from ONNX detector. |
| `PhysicalSemanticSegmenter.*` | Dense semantic class image. |
| `PhysicalInstanceSegmenter.*` | SAM-style prompted instance masks. |
| `PhysicalImageClassifier.*` | Whole-frame top-K classification. |
| `PhysicalPoseKeypointEstimator.*` | Per-entity keypoints. |
| `PhysicalSceneTextReader.*` | Scene OCR/text regions. |
| `PhysicalFacialExpressionDetector.*` | Face boxes + expression labels. |
| `PhysicalEntityTracker.*` | Pure-C++ track IDs and temporal smoothing. |

The default object detector asset is
`resources/models/vision/yolov8n_opencv_static.onnx`, a static 640x640 YOLOv8n
export simplified for OpenCV 4.11. The dynamic `yolov8n.onnx` export is kept for
provenance but is not the active Stage-2 model because OpenCV rejects its
`/model.12/Concat` node during import.

On Windows, local camera enumeration and automatic local-device opens use
`CAP_ANY` only. `CAP_DSHOW` / `CAP_MSMF` by-index probes are deliberately skipped
because this OpenCV 4.11 build can warn that those backends cannot capture by
index and has crashed during repeated startup probing.

Stage-2 operators receive `const cv::Mat&` model-space frames. They must not
store frame-bus `cv::Mat` headers beyond the route call.

`PhysicalPerceptionPrimitivesLoop` owns cadence/cache decisions. SAM instance
segmentation has an extra prompt-aware reuse gate: if the object detector result
is a cache hit and the quantized detector-box/class signature is unchanged, the
loop reuses the cached mask bundle instead of re-running SAM, with
`cache_reason=detector_prompt_cache`.

## Stage 2 Auxiliary — Human Interaction

| Files | Responsibility |
|---|---|
| `PhysicalInteractionLoop.*` | Non-blocking mainloop entry point and one-frame background worker for controller-oriented hand input. |
| `PhysicalHandGestureBackend.hpp` | Backend-neutral configuration, RGB input, and adapter interface. |
| `PhysicalMediaPipeHandGestureBackend.cpp` | Optional MediaPipe Tasks C adapter; a visible unavailable adapter is compiled when MediaPipe is off. |
| `PhysicalHandGestureBus.*` | Latest operational/result snapshot consumed by the UI and future controller layer. |
| `PhysicalHandGestureResult.hpp` | MediaPipe-free 21-landmark, handedness, gesture, provenance, and telemetry contract. |
| `PhysicalGestureControlLoop.*` | Temporal hysteresis, arming/cooldown safety, and allowlisted mouse/wake routing. |
| `PhysicalGestureControlConfigIO.*` | Schema-versioned offline binding persistence through the canonical runtime configuration. |
| `PhysicalGestureEventBus.*` | Bounded semantic `Started`/`Held`/`Released` event history for controller consumers. |
| `PhysicalGestureControlResult.hpp` | Backend-free controller event and operational-status contracts. |

The interaction branch consumes `PhysicalFrameBus` independently of the main
Stage-2 primitive bundle. It does not delay Stage 2/3/4/5, and Phase 1 does not
project gestures into model context. Its queue is deliberately bounded to one
frame: a new frame replaces stale pending work instead of allowing latency to
grow without bound.

The controller consumes completed interaction snapshots on the main thread. It never
injects raw per-frame labels directly: confidence hysteresis and dwell/release
timers first produce semantic gesture events. Computer actions are restricted
to the allowlisted binding table in `PhysicalGestureControlConfig`. The
Interaction UI can edit and persist label/action/phase/hold/cooldown/hand
mappings and provides a no-output dry-run mode. Pointer and mouse bindings are
always forced to require an armed control session.

## Stage 3 — Spatial Grounding

| Files | Responsibility |
|---|---|
| `PhysicalSpatialGroundingLoop.*` | Mainloop entry point; fuses frame + Stage-2 bundle with matching counters. |
| `PhysicalSpatialGroundingBus.*` | Latest spatial grounding snapshot. |
| `PhysicalSpatialGroundingResult.hpp` | Stage-3 result contract and telemetry. |
| `PhysicalMonocularDepthEstimator.*` | ONNX depth inference. |
| `PhysicalDepthMap.hpp` | Depth-map data contract. |
| `PhysicalSpatialGrounder.*` | Per-track range/surface/path-block/motion grounding. |
| `PhysicalVisualScaleFromDepth.*` | Promotes monocular VO scale using depth. |
| `PhysicalSpatialGroundingLogTag.hpp` | Stage-3 log tag. |

## Stage 5 — Localization / Mapping

| Files | Responsibility |
|---|---|
| `PhysicalLocalizationLoop.*` | Mainloop entry point; owns odometry, scale, and occupancy-grid updates. |
| `PhysicalLocalizationBus.*` | Latest localization snapshot. |
| `PhysicalLocalizationResult.hpp` | Pose/grid result contract. |
| `PhysicalVisualOdometer.*` | ORB + essential-matrix visual odometry. |
| `PhysicalOccupancyGridMapper.*` | Metric log-odds free-space grid. |
| `PhysicalLocalizationLogTag.hpp` | Stage-5 log tag. |

Stage 5 is numbered by dependency semantics: it runs before Stage 4 so the
world-state snapshot can include localization-derived facts.

## Stage 4 — World State / Reasoning Bridge

| Files | Responsibility |
|---|---|
| `PhysicalWorldStateLoop.*` | Mainloop entry point; builds the model-facing world snapshot. |
| `PhysicalWorldStateBus.*` | Latest identity-keyed world-state snapshot. |
| `PhysicalWorldStateResult.hpp` | Stage-4 result contract. |
| `PhysicalWorldStateBuilder.*` | Fuses Stage-2/3 results into stable entity state. |
| `PhysicalWorldStateContextProjector.*` | Projects world state into session/model context. |
| `PhysicalWorldStateMemoryWriter.*` | Writes durable world-state diffs to memory. |
| `PhysicalWorldStateLogTag.hpp` | Stage-4 log tag. |

## Adding or Moving Files

1. Pick exactly one stage/ownership row above.
2. Include via repository-root-relative paths from outside this folder, e.g.
   `#include "perception/physical/PhysicalFrameBus.hpp"`.
3. Keep local implementation includes minimal and direct.
4. Update this taxonomy and `docs/PHYSICAL_PERCEPTION_SUBSYSTEM.md` in the same
   change.
5. Build Release after any physical file move; the recursive CMake glob will
   find sources, but include paths still enforce the ownership boundary.
