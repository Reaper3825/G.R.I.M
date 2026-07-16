# MediaPipe Hand Gesture Integration — Phase 1

Phase 1 adds a controller-oriented hand interaction branch to GRIM's physical
perception layer. It does not add gestures to LLM context yet, and it does not
make MediaPipe a base dependency.

## Offline invariants

- `GRIM_USE_MEDIAPIPE_HAND_GESTURES` defaults to `OFF`.
- CMake has no `FetchContent`, package download, or model download path.
- The model must be a local filesystem path. URI-style model paths are rejected.
- There is no runtime network fallback.
- The opt-in build fails configuration unless
  `GRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED=ON` is supplied.
- That flag is an integrator attestation, not a blind switch: the supplied
  MediaPipe build must have its usage-metrics transport audited and disabled.
  A null `ca_bundle_path` alone is not treated as proof.
- The Physical Environment → Interaction tab reports the offline guarantee,
  metrics-transport status, backend/model state, errors, worker pressure,
  timings, frame provenance, and current hands.

MediaPipe's public privacy notice says task input is processed on-device, while
task performance/utilization metrics may be sent to Google. For that reason,
stock/unverified binaries are not accepted by GRIM's offline-only opt-in gate:
<https://github.com/google-ai-edge/mediapipe#privacy-notice>

## Runtime shape

`TickPhysicalInteraction()` pulls the latest immutable raw frame after Stage 1
and returns immediately. A dedicated worker owns the MediaPipe recognizer and a
single pending `FrameView`. If inference is behind, a new frame replaces the
pending frame; old work does not accumulate.

The worker uses MediaPipe Tasks C `GestureRecognizer` in `VIDEO` mode, allowing
its temporal hand tracking to reduce repeated detection work. Results are copied
into `PhysicalHandGestureSnapshot`, which contains no MediaPipe handles or
types, and are published through `PhysicalHandGestureBus`.

Phase 1's UI overlays landmarks only when the result counter exactly matches
the displayed raw frame. A stale result remains visible in diagnostics, but its
geometry is withheld so the UI never presents a misleading overlay.

## Opt-in configuration

The adapter targets the MediaPipe Tasks C API recorded in
`GRIM_MEDIAPIPE_VERSION` (default `0.10.35`). Supply an already-built local
library and its matching source/install root:

```powershell
cmake -S . -B build `
  -DGRIM_USE_MEDIAPIPE_HAND_GESTURES=ON `
  -DGRIM_MEDIAPIPE_ROOT=D:/deps/mediapipe-0.10.35 `
  -DGRIM_MEDIAPIPE_TASKS_C_LIBRARY=D:/deps/mediapipe/lib/mediapipe_tasks_c.lib `
  -DGRIM_MEDIAPIPE_VERSION=0.10.35 `
  -DGRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED=ON
```

Place the local task model at:

`resources/models/vision/mediapipe/gesture_recognizer.task`

No model is committed or fetched by this phase. A different local path can be
provided later through `RequestConfigurePhysicalHandGestures()`.

## Phase 1 acceptance checks

1. With MediaPipe disabled, GRIM configures as before and the Interaction tab
   reports `Backend unavailable` without affecting camera or other stages.
2. With MediaPipe enabled but no model, the tab reports `Model missing`; no
   network request occurs.
3. With an audited library and local model, state reaches `Ready`, frame counters
   advance, and gesture/landmark results appear.
4. Under load, `Queue replacements` may rise but mainloop latency does not track
   inference latency.
5. Disabling gestures stops inference while the rest of physical perception
   continues.

Before shipping an enabled build, test it behind an outbound-deny firewall and
capture process network activity during initialization, inference, and shutdown.
