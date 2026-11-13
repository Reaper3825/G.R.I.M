# Perception provider — API v9

The DLL owns digital and physical acquisition, OCR/native UI primitives,
detection/segmentation/classification/pose/face/text/tracking operators,
calibration, gesture recognition, spatial grounding, localization, world-state
fusion, internal buses, and their worker/model/device lifetimes.

## Breaking boundary

No host perception implementation is linked. `PerceptionContextManager`, the
legacy capture/OCR/vision wrappers and their detached initialization are removed.
No compatibility service, automatic question routing, session projection,
automatic durable-memory writes, device-server pointer, environment-panel
factory, gesture input/wake dispatch, or host model-registry wiring remains.
The old three perception tool names and `click_on` are retired. Pure targeting
and input-device primitives remain under `core/input`.

The digital/physical primitives are preserved as provider-private implementations.
C++/OpenCV objects stay inside the DLL. The public boundary is copied structured
JSON tools plus a `perception.observations` resource. UI/session/memory integrations
must be new consumers, not direct references to these implementation classes.

## Lifecycle

Load stages tools, an idle observation resource and a generic host timer. It does
not acquire a camera/screen or load an inference model. Explicit tools start
capture. The provider owns processing policy; generic timer callbacks pump the
physical stages, preserving calibration gating and source-frame coherence.
Digital capture/primitives retain their joined workers. Shutdown cancels the
timer, joins analysis/capture workers and releases engines and frame caches.
Stop/restart resets shutdown flags; there are no detached plugin workers.

## Tools

All tools accept JSON objects and return structured output. They are Debug tools.

| Tool | Arguments / behavior |
|---|---|
| `perception.observe` | `{}`: latest digital OCR/UI observations, physical primitive state/detections/text, gesture landmarks and world entities; unavailable observations are explicit. Does not capture or persist memory. |
| `perception.digital.start` | `mode`: `active-monitor` (default), `active-window`, `virtual-desktop`, or `monitor` with `monitor_index`; `interval_ms`: 33–60000 (default 1000); `ocr` and `automation` booleans. |
| `perception.digital.stop` | `{}`: stop/join digital workers and clear their buses. |
| `perception.physical.open` | `url` and optional `label`: open an explicit camera source, e.g. `device:0`; alternatively `left_url` and `right_url` open stereo sources, with optional `left_label`, `right_label`, and `maximum_pair_skew_ms`. Replacing an active camera resets the physical pipeline. |
| `perception.physical.close` | `{}`: stop physical stages and close devices. |
| `perception.configure` | `operator` plus object-valued `config`. Config replacement is explicit and not automatically persisted. |
| `perception.calibrate` | `action`: `automatic` (default), `start`, `stop`, `sample`, `solve`, `save`, `load`, `clear`, or `pattern` (with `cols`, `rows`, `square_meters`); requires an open physical camera. |

Operator identifiers: `object_detector`, `semantic_segmenter`, `image_classifier`,
`pose_estimator`, `scene_text_reader`, `facial_expression`, `face_recognizer`,
`entity_tracker`, `instance_segmenter`, `depth`, `hand_gestures`, `conditioning`,
`spatial_grounder`, `odometry`, `occupancy_grid`, and `world_state`.
Supported scalar config fields are enumerated in `perception_config.cpp`; unknown
fields are rejected by the model-operator adapters. Empty model paths mean
NoModelConfigured. Existing ai_config/MMO vision entries no longer load models.
No automatic downloads, dependency installation, or model-server startup occurs.

Digital observations carry frame provenance, desktop/source rectangles, OCR
regions and confidence, and UI-control bounds/target-correlation status. Physical
primitive snapshots carry their frame ID and operator status/errors; world-state
snapshots carry their own frame ID. Consumers must not assume asynchronous stage
snapshots describe the same frame. Dense masks/depth and detailed operator results
remain inside the provider, not serialized as pixels in the resource.

## Build and validation

Use existing dependencies; deploy only with a matching v9 host and plugins.

```powershell
cmake -S plugins/perception_plugin -B .codex-build/perception-owned -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF -DGRIM_PERCEPTION_PROVIDER_TESTS=ON
cmake --build .codex-build/perception-owned --config Release --target perception_plugin perception_provider_tests --parallel 4
# Ensure installed native dependency DLLs are available on PATH.
ctest --test-dir .codex-build/perception-owned -C Release --output-on-failure
```

The provider test loads the real DLL, rejects old ABI, checks idle/no-observation
state, malformed requests, failed initialization, repeated shutdown and reload.
It does not open devices or execute inference. Separate input/registry lifecycle
tests and pure click-target geometry tests cover the host decoupling.

MediaPipe remains optional and uses existing local audited binaries when enabled.
Absent support is reported by its primitive backend. OpenCV, Tesseract and ONNX
Runtime are linked by the provider; no GRIM host symbols are imported.
