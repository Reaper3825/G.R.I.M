# Physical Perception Subsystem

> Location: [perception/physical/](../perception/physical)  
> File taxonomy: [perception/physical/README.md](../perception/physical/README.md)
> Namespace: `GRIM::Perception::Physical`
> Mainloop integration: [main.cpp](../main.cpp) lines ~593–614

The physical perception subsystem is GRIM's pipeline for understanding the
**real-world camera feed** (USB webcam, builtin camera, or networked IP/RTSP
camera). It is intentionally separate from the on-screen "digital" perception
(`perception/perception.cpp`, `vision_ai.cpp`) which deals with what is
displayed on the user's monitor.

The subsystem is structured as a **strict 5-stage pipeline**, where each
stage:

1. Has **one** mainloop entry point of the form `Tick<Stage>()`.
2. Lazy-initialises on the first tick.
3. Reads from the bus(es) of upstream stage(s) and writes to its own bus.
4. Exposes thread-safe `Request*` mutators for the UI.
5. Follows project Rule 20 — every failure path is loud (throws or sets a
   `last_error_reason` the UI surfaces verbatim). No silent fallbacks.

---

## 1. Pipeline at a Glance

```
                ┌───────────────────────────────────────────────────────────┐
                │                       Mainloop                             │
                │   (one frame per iteration; calls Ticks in fixed order)   │
                └───────────────────────────────────────────────────────────┘
                                            │
                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  STAGE 1  PhysicalEnvironmentLoop      → PhysicalFrameBus                │
  │           (camera I/O, conditioning, scene-stability)                    │
  └──────────────────────────────────────────────────────────────────────────┘
                                            │  raw_image, model_image, metadata
                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  STAGE 2  PhysicalPerceptionPrimitivesLoop                               │
  │           → PhysicalPerceptionPrimitiveBus                               │
  │           (det / seg / classify / pose / OCR / face / tracker /          │
  │            instance-seg / class-policy)                                  │
  └──────────────────────────────────────────────────────────────────────────┘
                                            │  detections, masks, tracks…
                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  STAGE 3  PhysicalSpatialGroundingLoop  → PhysicalSpatialGroundingBus    │
  │           (monocular depth + per-track grounding: range, surface,        │
  │            motion, path-block)                                           │
  └──────────────────────────────────────────────────────────────────────────┘
                                            │  depth map + grounded entities
                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  STAGE 5  PhysicalLocalizationLoop      → PhysicalLocalizationBus        │
  │           (visual odometry + occupancy grid; needs calibration)          │
  └──────────────────────────────────────────────────────────────────────────┘
                                            │  T_world_camera, trajectory, grid
                                            ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  STAGE 4  PhysicalWorldStateLoop        → PhysicalWorldStateBus          │
  │           (single identity-keyed snapshot the model reads)               │
  └──────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                                  Reasoning / Model context
```

> **Stage numbering.** Stages are **numbered by data dependency**, not by
> tick order. Stage 4 (world state) ticks **after** Stage 5 (localization)
> so future revisions can stamp `T_world_camera` onto every entity in a
> single snapshot. The mainloop call order is:
> `Environment → Interaction → PerceptionPrimitives → SpatialGrounding → Localization → WorldState`.
> Interaction is an auxiliary `PhysicalFrameBus` consumer, not a dependency of
> the numbered world-understanding stages.

---

## 2. Cross-Cutting Contracts

These rules apply to every stage and every result struct.

### 2.1 Coordinate spaces

There are **two** pixel spaces, and they are **not** related by a simple
scale because the conditioner letterboxes.

| Name             | Origin / size               | Use                                                  |
| ---------------- | --------------------------- | ---------------------------------------------------- |
| **RAW** space    | Original sensor frame       | Drawing overlays on the unmodified video             |
| **MODEL** space  | Conditioned/letterboxed     | What every model and operator actually sees          |

Conversion is the affine `PhysicalSignalRawToModelTransform`
(`scale_x, scale_y, offset_x, offset_y`) carried in
`PhysicalFrameMetadata`. Every result struct **stores both `model_box`
and `raw_box` explicitly**. **Consumers MUST NOT re-derive one from the
other** — the letterbox padding makes the relationship affine-with-offset,
not a pure scale.

### 2.2 Frame counters

Every published snapshot carries a `source_frame_counter` (and where
relevant `source_perception_results_counter`,
`source_grounding_results_counter`). Downstream stages **only act when
counters advance and match** — this prevents fusing a depth map for
frame N with detections from frame N-1.

### 2.3 Time

`steady_clock` (monotonic) is the source of truth for dt and rate
calculations. Wall clock is recorded only for human-readable display.

### 2.4 Failure handling (Rule 20)

* Hard configuration / I/O errors → **throw**.
* Expected transient conditions (no frame yet, lost tracking, no model
  loaded) → set `state` enum + `last_error_reason` string. The UI
  displays the reason verbatim. No silent fallback ever runs.

### 2.5 Threading

* Every `Tick*()` is mainloop-only.
* Every `Request*()` and `Get*Snapshot()` is thread-safe (mutex inside).
* Operators are **not** internally thread-safe; the loops own them and
  hold a mutex around every call.
* `TickPhysicalInteraction()` is a non-blocking exception by design: it only
  replaces a one-frame queue. Its loop-owned worker performs BGR-to-RGB
  conversion and MediaPipe inference, then publishes a pure-data snapshot.

### 2.6 File taxonomy

`perception/physical/README.md` is the local ownership map for this folder.
When adding or moving a physical-perception file, assign it to exactly one
taxonomy row there and update the row in the same change. This prevents the
flat `Physical*.{hpp,cpp}` namespace from turning into a junk drawer with a
camera attached — delightful image, bad architecture.

### 2.7 Frame packet ownership

`PhysicalFrameBus::FrameView` now pins an immutable `PhysicalFramePacket`.
Its `raw_image`, `model_image`, and `image` fields are shallow `cv::Mat`
headers into the packet, not per-consumer pixel copies. Consumers MUST treat
them as read-only and `clone()` into local scratch before any in-place OpenCV
mutation.

---

## 3. Stage 1 — `PhysicalEnvironmentLoop`

**Files:** [PhysicalEnvironmentLoop.hpp](../perception/physical/PhysicalEnvironmentLoop.hpp), [PhysicalCameraDirectory.hpp](../perception/physical/PhysicalCameraDirectory.hpp), [PhysicalCameraSource.hpp](../perception/physical/PhysicalCameraSource.hpp), [PhysicalCameraStream.hpp](../perception/physical/PhysicalCameraStream.hpp), [PhysicalFrameConditioner.hpp](../perception/physical/PhysicalFrameConditioner.hpp), [PhysicalFrameBus.hpp](../perception/physical/PhysicalFrameBus.hpp), [PhysicalSceneStability.hpp](../perception/physical/PhysicalSceneStability.hpp), [PhysicalNicScan.hpp](../perception/physical/PhysicalNicScan.hpp).

**Responsibility:** turn a chosen camera URL into a stream of conditioned
BGR frames on `PhysicalFrameBus`.

### Components

| Class                                     | Job                                                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `PhysicalNicScan` (per-OS .cpp/.mm)       | Enumerate IPv4 NICs (loopback + link-local marked `Disabled`).                                       |
| `PhysicalCameraDirectory`                 | Combine local NICs + local USB/builtin/virtual cameras + hub devices (DeviceCommServer) into a candidate list; local rows are backend-pinned so multiple cameras on one host are selectable. |
| `PhysicalCameraSource`                    | One row in the directory: origin, status, label, `url_template`.                                     |
| `PhysicalCameraStream`                    | OpenCV `VideoCapture` worker; reports state Idle / Connecting / Streaming / Failed.                  |
| `PhysicalFrameConditioner`                | Resize (Stretch or Letterbox), denoise (median blur), exposure correction, optional deblur/stabilization, color-space, **quality gate**, and per-frame `PhysicalSceneStability` thumbnail diff. |
| `PhysicalFrameBus`                        | Singleton single-producer/multi-consumer latest-frame slot (raw + model + metadata).                |

### Per-frame metadata (`PhysicalFrameMetadata`)

* `capture_steady_ns`, `publish_steady_ns`, `capture_wall_ns`
* `raw_width/raw_height`, `model_width/model_height`
* `raw_to_model` — the affine transform (authoritative; never re-derive)
* `color_space_label`, `pipeline_summary`, `applied_exposure_gain`
* `scene_stability` — used by every cache-aware downstream operator

### Quality gate

Frames are **dropped** at the producer (with a logged reason) when:

* Mean luma `< min_mean_luma` (black startup frames)
* Mean luma `> max_mean_luma` (lens cap removed mid-stream / blown-out)
* Clipped-pixel ratio `> max_clipped_pixel_ratio`
* Laplacian variance `< min_laplacian_variance` (motion smear)

This prevents downstream temporal context from being poisoned.

### UI surface

```cpp
RegisterPhysicalEnvironmentDeviceServer(server);   // optional, for hub enum
RequestPhysicalCameraDirectoryRefresh();           // re-scan NICs + hub
RequestOpenPhysicalCameraSource(url, label);
RequestClosePhysicalCameraSource();
RequestConfigurePhysicalSignalConditioning(cfg);   // atomic swap
RequestResetPhysicalSignalConditioningDefaults();
GetPhysicalCameraDirectorySnapshot();
GetActiveStreamFps(); GetActiveStreamFrameCounter();
GetLastEnvironmentError();
```

### Camera calibration

Calibration is its own subsystem layered on top of Stage 1:

* [PhysicalCameraCalibrator.hpp](../perception/physical/PhysicalCameraCalibrator.hpp) — state machine (Uncalibrated → LoadedFromDisk → Capturing → Calibrated → Failed), pulls frames from `PhysicalFrameBus`, runs lighting-adaptive checkerboard detection, manages a coverage-aware sample pool, runs `cv::calibrateCamera`, and exposes `UndistortBgrFrameUsingPhysicalCalibration()`.
* [PhysicalCalibrationPattern.hpp](../perception/physical/PhysicalCalibrationPattern.hpp) — physical pattern descriptor (cols × rows × square size in metres).
* [PhysicalCalibrationStore.hpp](../perception/physical/PhysicalCalibrationStore.hpp) — disk persistence for `K` and distortion coefficients.

The calibrator is consumed by **Stage 5** (visual odometry refuses to run
without intrinsics).

---

## 4. Stage 2 — `PhysicalPerceptionPrimitivesLoop`

**Files:** [PhysicalPerceptionPrimitivesLoop.hpp](../perception/physical/PhysicalPerceptionPrimitivesLoop.hpp), [PhysicalPerceptionPrimitiveBus.hpp](../perception/physical/PhysicalPerceptionPrimitiveBus.hpp), [PhysicalPerceptionPrimitiveResult.hpp](../perception/physical/PhysicalPerceptionPrimitiveResult.hpp), per-operator headers (below), [PhysicalImageOperatorState.hpp](../perception/physical/PhysicalImageOperatorState.hpp).

**Responsibility:** run every enabled per-frame vision operator on the
latest `FrameBus` frame and publish the **coherent** result set.

### Operators

| Operator                            | Backend                              | Output                                                                  |
| ----------------------------------- | ------------------------------------ | ----------------------------------------------------------------------- |
| `PhysicalObjectDetector`            | `cv::dnn::Net` ONNX (Ultralytics YOLOv8 layout) | Axis-aligned `PhysicalObjectDetection[]` with `model_box` + `raw_box`. |
| `PhysicalSemanticSegmenter`         | ONNX                                 | `class_id_image` (CV_32SC1) at model resolution.                        |
| `PhysicalInstanceSegmenter`         | SAM 2 ONNX, prompted by detector boxes | `PhysicalInstanceMask[]` packed inside tight `mask_model_bbox`s.       |
| `PhysicalImageClassifier`           | ONNX                                 | Top-K class scores.                                                     |
| `PhysicalPoseKeypointEstimator`     | ONNX                                 | Per-instance keypoints in MODEL + RAW space.                            |
| `PhysicalSceneTextReader`           | ONNX (detector + recognizer)         | Quad polygons + UTF-8 strings.                                          |
| `PhysicalFacialExpressionDetector`  | ONNX                                 | Per-face expression class + score.                                      |
| `PhysicalEntityTracker`             | Pure C++ (no model)                  | Identity-keyed `PhysicalEntityTrack[]`; greedy IoU + class-gated assignment, exponential smoothing in MODEL space, monotonic never-reused track IDs. |
| `PhysicalClassPolicy`               | Pure C++                             | Applies merge map + priority cutoff to per-operator results in place; emits ranked summary. |

`PhysicalObjectDetector` is wired to `resources/models/vision/yolov8n_opencv_static.onnx`.
That file is a static `1x3x640x640` simplified export of YOLOv8n. Do **not**
replace it with the dynamic opset-22 `yolov8n.onnx` export unless OpenCV import
is re-validated first; OpenCV 4.11 rejects that graph at `/model.12/Concat`.

On Windows, local camera directory probing uses `CAP_ANY` rather than probing
`CAP_DSHOW` / `CAP_MSMF` by numeric index. This is intentional: OpenCV 4.11 can
report those backends as available and then reject by-index capture with native
warnings, and repeated startup probes have caused access violations. The Stage-1
lazy init also reuses any directory already loaded by the UI instead of probing
the same cameras twice during startup.

### Operator state machine (`PhysicalImageOperatorState`)

`NoModelConfigured → ModelLoading → ModelLoaded → InferenceFailed`.
The Loop only routes frames to operators in `ModelLoaded` whose enable
flag is `true`. `NoModelConfigured` is the **default** — paths are
optional and the UI surfaces the state explicitly.

### Per-tick algorithm

1. `PullLatestFrameView`. Skip if counter has not advanced.
2. For each enabled, model-loaded operator: `RouteFrameTo<Operator>()`.
3. Run the tracker on the (possibly-empty) detector output.
4. Run the instance segmenter prompted by detector boxes. The Stage-2 loop
   reuses the cached SAM mask bundle when the detector itself reused cached
   detections and the quantized prompt signature is unchanged
   (`cache_reason=detector_prompt_cache`); zero prompts remain a cheap no-op.
5. Apply class policy in place.
6. Publish the whole bundle to `PhysicalPerceptionPrimitiveBus` once.

### UI surface

```cpp
RequestSetPhysicalPerceptionPrimitivesEnableFlags(flags);
RequestConfigurePhysicalObjectDetector(cfg);
RequestConfigurePhysicalSemanticSegmenter(cfg);
RequestConfigurePhysicalImageClassifier(cfg);
RequestConfigurePhysicalPoseKeypointEstimator(cfg);
RequestConfigurePhysicalSceneTextReader(cfg);
RequestConfigurePhysicalFacialExpressionDetector(cfg);
RequestConfigurePhysicalEntityTracker(cfg);
RequestConfigurePhysicalInstanceSegmenter(cfg);
RequestConfigurePhysicalClassPolicy(cfg);
```

---

## 5. Stage 3 — `PhysicalSpatialGroundingLoop`

**Files:** [PhysicalSpatialGroundingLoop.hpp](../perception/physical/PhysicalSpatialGroundingLoop.hpp), [PhysicalSpatialGroundingBus.hpp](../perception/physical/PhysicalSpatialGroundingBus.hpp), [PhysicalSpatialGroundingResult.hpp](../perception/physical/PhysicalSpatialGroundingResult.hpp), [PhysicalMonocularDepthEstimator.hpp](../perception/physical/PhysicalMonocularDepthEstimator.hpp), [PhysicalSpatialGrounder.hpp](../perception/physical/PhysicalSpatialGrounder.hpp), [PhysicalDepthMap.hpp](../perception/physical/PhysicalDepthMap.hpp), [PhysicalVisualScaleFromDepth.hpp](../perception/physical/PhysicalVisualScaleFromDepth.hpp).

**Responsibility:** turn each tracked 2D entity into a structured
spatial fact.

### Components

* **`PhysicalMonocularDepthEstimator`** — ONNX backbone (e.g.
  Depth-Anything / MiDaS), produces a `PhysicalDepthMap`. Units may be
  `Relative` (0.0 = nearest, 1.0 = farthest) or `Meters` (when a metric
  model is loaded).
* **`PhysicalSpatialGrounder`** — pure C++. For every track:
  * `range_value` = robust median over the box interior of the depth map
  * `support_surface` ∈ `{Unknown, Floor, Table, Wall}` from depth-vs-image-y profile
  * `path_block_score ∈ [0,1]` and boolean
  * `motion_state` ∈ `{Unknown, Static, Moving, Coasted}` from 2D + depth velocity
  * `moved_since_last_frame`

### Per-tick algorithm

1. Pull latest frame from `PhysicalFrameBus`.
2. Pull latest perception bundle from `PhysicalPerceptionPrimitiveBus`.
3. **Only proceed if both advanced AND `source_frame_counter` matches.**
4. Depth pass → grounder pass → publish to `PhysicalSpatialGroundingBus`.

### Numerical contract

Every magic constant lives in `PhysicalSpatialGrounderConfig` — the
result struct contains only **derived** values and the inputs that
produced them.

---

## 6. Stage 5 — `PhysicalLocalizationLoop`

**Files:** [PhysicalLocalizationLoop.hpp](../perception/physical/PhysicalLocalizationLoop.hpp), [PhysicalLocalizationBus.hpp](../perception/physical/PhysicalLocalizationBus.hpp), [PhysicalLocalizationResult.hpp](../perception/physical/PhysicalLocalizationResult.hpp), [PhysicalVisualOdometer.hpp](../perception/physical/PhysicalVisualOdometer.hpp), [PhysicalOccupancyGridMapper.hpp](../perception/physical/PhysicalOccupancyGridMapper.hpp).

**Responsibility:** answer **WHERE** the camera is and what free space
it has traversed.

### Why it exists

Stages 1–4 are inherently **ego-centric and per-frame** (every
coordinate is in the current image). That is enough for "what do I see
right now" but **not** enough for an agent that moves (robot, drone, AR
glasses, phone, laptop being carried). Localization gives the model:

1. A persistent 6-DoF pose `T_world_camera` (4×4 `CV_64F`).
2. A growing 2D log-odds occupancy grid (Nav2 / Thrun §9.2 convention).

### `PhysicalVisualOdometer`

ORB → BFMatcher (Hamming, cross-check) → `cv::findEssentialMat` (RANSAC
with intrinsics from `PhysicalCameraCalibrator`) → `cv::recoverPose` →
`T_world_curr = T_world_prev * inv([R|t])`.

States: `Uninitialized → Initializing → Tracking → Lost / Failed`.

**Scale ambiguity is explicit** in
`PhysicalLocalizationPoseScaleState`:

| State                | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `Unknown`            | No pose yet                                          |
| `UnscaledMonocular`  | Translation has unit norm — **not metric**           |
| `ScaledByDepthMap`   | Monocular VO + monocular depth → metric (drifts)     |
| `ScaledByStereo`     | (future) stereo / RGB-D                              |
| `ScaledByImu`        | (future) VIO-fused                                   |

`PhysicalVisualScaleFromDepth` is the bridge that promotes
`UnscaledMonocular` → `ScaledByDepthMap` when a depth map is available.

### `PhysicalOccupancyGridMapper`

* Marks the camera's current cell as **free** (subtracts
  `free_log_odds_increment`) and Bresenham-rasterises a corridor of free
  cells from the previous to current pose.
* Only updates when `pose_scale_state` is metric — refuses to silently
  produce a meaningless map from unit-norm translations.
* Cells clamped to `[-max_abs_log_odds, +max_abs_log_odds]`.
* Does **not** mark "occupied" yet — that requires a calibrated depth
  ray-cast (planned `RouteDepthToPhysicalOccupancyGridMapper()`).

### Per-tick algorithm

1. Lazy-init odometer + grid mapper.
2. Pull camera intrinsics from `PhysicalCameraCalibrator`. **No
   intrinsics → publish a `Failed` snapshot every tick with a clear
   reason.** (Rule 20: do not pretend to track without calibration.)
3. Pull latest frame from `PhysicalFrameBus`.
4. Run odometer → update grid → publish `PhysicalLocalizationSnapshot`.

### UI surface

```cpp
RequestConfigurePhysicalLocalizationOdometer(cfg);
RequestConfigurePhysicalLocalizationGrid(cfg);
RequestResetPhysicalLocalization();   // drop world frame + grid (teleport)
```

---

## 7. Stage 4 — `PhysicalWorldStateLoop`

**Files:** [PhysicalWorldStateLoop.hpp](../perception/physical/PhysicalWorldStateLoop.hpp), [PhysicalWorldStateBus.hpp](../perception/physical/PhysicalWorldStateBus.hpp), [PhysicalWorldStateResult.hpp](../perception/physical/PhysicalWorldStateResult.hpp), [PhysicalWorldStateBuilder.hpp](../perception/physical/PhysicalWorldStateBuilder.hpp).

**Responsibility:** the **single bridge between raw perception and
GRIM's reasoning**. Fuses Stage-2 + Stage-3 outputs into one
deterministic, identity-keyed snapshot.

### Snapshot fields (per entity, keyed by `track_id` = `object_id`)

| Field                          | Source                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| `class` / `confidence`         | Latest matched detection                                   |
| `model_box` / `raw_box` / centre | Smoothed tracker state (both spaces)                     |
| `velocity_2d` (px/sec) + `velocity_depth` (units/sec) | Tracker + grounder        |
| `visibility`                   | `Visible / Occluded / Coasting / Unknown` from depth ordering + box overlap |
| `text_on_object`               | OCR lines whose centroid lies inside the box               |
| `last_seen_time`               | `steady_clock` ns + frame counter                          |
| `relations` (top-K)            | `Contains, ContainedBy, Overlaps, LeftOf, RightOf, Above, Below, NearerThan, FartherThan` |

### Auditability

The snapshot carries `source_frame_counter`,
`source_perception_results_counter`, and
`source_grounding_results_counter` so any consumer can audit which
inputs produced it. Relations are sorted by `other_object_id` ASC for
stable diffing across frames.

### Numerical contract

* Every coordinate stored in **both** model and raw space.
* No magic constants in the result — `PhysicalWorldStateBuilderConfig`
  owns thresholds (occlusion overlap, depth velocity threshold, etc.).

---

## 8. Bus Pattern

Every stage owns one bus. Buses are singletons, single-producer,
multi-consumer, mutex-protected, and follow the same shape:

```cpp
class PhysicalXxxBus {
public:
    static PhysicalXxxBus& Instance();
    void PublishPhysicalXxxToBus(const PhysicalXxxSnapshot& snap);
    bool PullLatestPhysicalXxx(PhysicalXxxSnapshot& out, uint64_t& last_seen_counter) const;
    bool HasEverPublishedPhysicalXxx() const;
    void ResetPhysicalXxxBus();
};
```

`PullLatest*` returns `true` **only when the snapshot's counter is
strictly newer** than `last_seen_counter`, and updates the caller's
counter in that case. This is the mechanism that prevents redundant
work downstream and lets the UI poll cheaply at any rate.

---

## 9. Logging

Every stage has a dedicated log tag header:

* [PhysicalEnvironmentLogTag.hpp](../perception/physical/PhysicalEnvironmentLogTag.hpp)
* [PhysicalPerceptionPrimitivesLogTag.hpp](../perception/physical/PhysicalPerceptionPrimitivesLogTag.hpp)
* [PhysicalSpatialGroundingLogTag.hpp](../perception/physical/PhysicalSpatialGroundingLogTag.hpp)
* [PhysicalLocalizationLogTag.hpp](../perception/physical/PhysicalLocalizationLogTag.hpp)
* [PhysicalWorldStateLogTag.hpp](../perception/physical/PhysicalWorldStateLogTag.hpp)

Use these tags so the runtime log filter can isolate one stage at a
time during debugging.

### Windows ONNX model setup

The Stage-2/Stage-3 vision operators load the ONNX paths declared in
`ai_config.json` under `mmo.sub_models`. On Windows, startup checks the
configured model files before wiring the operators. If any required file
is missing or empty, GRIM runs `scripts/setup_windows_physical_vision.ps1`
automatically, then re-validates the paths before continuing. The script
populates `resources/models/vision` with the detector, segmenter, OCR,
face, emotion, SAM2, MobileCLIP, and Depth-Anything files expected by the
runtime config.

Model entries may set `enabled=false` to keep downloaded weights in the
registry without wiring them into the live OpenCV operators. This is used
for `depth_anything_v2_metric_indoor_small_518`: OpenCV 4.11 rejects that
ONNX export at the backbone `Resize` node, so Stage 3 keeps the
OpenCV-compatible `midas_v21_small_256` entry active instead.

You can still run `scripts/setup_windows_physical_vision.ps1` manually
from the repo root to pre-warm a fresh checkout or refresh corrupted
files with `-Force`.

The C++ bootstrap resolves these config paths against `getGrimRootDir()`
before handing them to OpenCV or ONNX Runtime. That keeps the models
loadable when `GRIM.exe` is launched from Visual Studio, PowerShell, or a
build output directory instead of relying on the process working
directory.

---

## 10. Adding a New Operator (Recipe)

1. **Result type** — add a struct to
   [PhysicalPerceptionPrimitiveResult.hpp](../perception/physical/PhysicalPerceptionPrimitiveResult.hpp)
   (or a new result header for a new stage). Store every coordinate in
   **both** model and raw space.
2. **Operator class** — header + .cpp under `perception/physical/`.
   Owns its model, exposes `PhysicalImageOperatorState`,
   `Load…(cfg)`, `RouteFrameTo…(...)`, `Reset…()`. Throw on hard
   errors; populate `last_error_reason` on inference failure.
3. **Loop integration** — add an enable flag to the loop's
   `EnableFlags` struct, route the frame in `Tick…()`, and add a
   `RequestConfigure…()` mutator.
4. **Bus** — extend the relevant bus snapshot to carry the new field,
   keeping `source_frame_counter` correlation intact.
5. **UI** — surface state + last_error_reason verbatim. **Never**
   silently hide a `NoModelConfigured` or `InferenceFailed` operator.

---

## 11. Quick Reference — File Map

The authoritative per-file taxonomy lives in
[`perception/physical/README.md`](../perception/physical/README.md). Use that
file for ownership decisions; this architecture doc describes behavior and
stage contracts.

---

## 12. Invariants Worth Memorising

1. **`Tick*()` is called exactly once per mainloop iteration, in this
   order:** Environment → Interaction → PerceptionPrimitives →
   SpatialGrounding → Localization → WorldState.
2. **No file outside a stage's loop should touch that stage's operators
   directly.** All mutation goes through `Request*()`.
3. **Never re-derive raw coordinates from model coordinates** (or vice
   versa). Use the stored value or the `raw_to_model` transform.
4. **Never act on a downstream snapshot whose `source_frame_counter`
   does not match the upstream snapshot you intend to fuse with.**
5. **Failures are loud.** Throw on hard errors; populate
   `last_error_reason` on transient ones; never silently fall back.
6. **Localization without calibration publishes `Failed`** — it does
   not estimate intrinsics on the fly.
7. **Occupancy grid only updates with metric scale.** A free-space map
   from unit-norm monocular VO would be a lie.
