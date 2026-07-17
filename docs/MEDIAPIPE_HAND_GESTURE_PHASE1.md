# MediaPipe Hand Gesture Integration — Phase 1

> **Status: COMPLETE / READY** — The base hand-perception layer and its
> Physical Environment UI visuals were validated in the live GRIM runtime on
> July 16, 2026.

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

## Completion validation

The locally built MediaPipe backend loaded successfully and reached `Ready` in
the running application. Live-camera validation covered:

- Multiple supported hand poses across repeated recognition changes.
- Left-hand-only and right-hand-only recognition.
- Simultaneous left-and-right-hand recognition.
- Correct live presentation of handedness, gesture results, and hand landmark
  visuals in the Physical Environment Interaction UI.
- Continued UI and mainloop responsiveness during live recognition, with
  inference timing, frame provenance, counters, and queue pressure visible in
  the diagnostics surface.

The base-layer UI visuals are therefore ready for use. Interactive runtime
performance passed the live test without an observed UI or mainloop stall. No
fixed-camera numeric FPS or latency benchmark was recorded in this validation,
so future performance comparisons should use the UI's timing and queue-pressure
fields rather than treating this completion record as a hardware benchmark.

This completion applies to local hand recognition, result publication, and UI
visibility. Phase 2 computer-control routing and any future projection of hand
state into LLM context remain separate validation scopes.

The later offline binding-customizer implementation is documented separately
in [`MEDIAPIPE_GESTURE_CONTROL_CUSTOMIZER_PHASE1.md`](MEDIAPIPE_GESTURE_CONTROL_CUSTOMIZER_PHASE1.md).

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
