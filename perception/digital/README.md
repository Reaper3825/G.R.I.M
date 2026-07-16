# Digital Perception: Capture Spine and Frame-Coherent Primitives

Namespace: `GRIM::Perception::Digital`

This directory owns screen acquisition, OCR, optional native UI inspection, and
the first reasoning handoff. It does not own multimodal inference, durable
memory writes, or input control.

## Runtime flow

1. `DigitalEnvironmentLoop` captures at a configured cadence on a worker thread.
2. `DigitalCaptureSource` resolves the active monitor, selected monitor, active
   window, or virtual desktop from the live operating-system display directory.
3. `DigitalFrameBus` publishes one immutable `DigitalFramePacket` per attempt.
   Failed attempts are published too, with an empty image and explicit status.
4. `DigitalPerceptionPrimitivesLoop` consumes the exact immutable frame on a
   separate worker. Cross-platform OCR always has first-class status. An
   optional host-native automation provider can add structured controls.
5. `DigitalPerceptionPrimitiveBus` publishes one coherent OCR + automation
   snapshot carrying the source frame counter.
6. `DigitalContextProjector` pulls capture and primitive snapshots and updates
   `SessionContextManager::VisualContext::DigitalVisual` for the default session.
7. `RouterMetadataBuilder` exposes both OCR and structured controls to reasoning.
8. `UIDigitalEnvironmentPanel` consumes the same buses for human-visible preview
   and telemetry. It never captures pixels itself.

All analyzers and the Digital Environment panel consume `DigitalFrameBus`. They
must never recapture the desktop independently.

## Files

| File | Ownership |
|---|---|
| `DigitalCaptureTypes.*` | Capture requests, monitor descriptors, metadata, statuses, device origin, and string conversion. |
| `DigitalCaptureSource.*` | Platform source factory and Win32 capture implementation. |
| `DigitalFrameBus.*` | Immutable latest-attempt handoff with per-consumer cursors. |
| `DigitalEnvironmentLoop.*` | Worker lifecycle, cadence, counters, and capture health. |
| `DigitalPerceptionPrimitiveTypes.*` | Portable OCR-region, UI-element, status, and provenance contracts. |
| `DigitalOcrProvider.*` | Cross-platform OCR provider contract and local Tesseract implementation. |
| `DigitalAutomationProvider.*` | Optional platform-native structure provider; Windows UI Automation today. |
| `DigitalPerceptionPrimitivesLoop.*` | Independent frame consumer and provider orchestration worker. |
| `DigitalPerceptionPrimitiveBus.*` | Immutable latest semantic snapshot with frame-counter coherence. |
| `DigitalContextProjector.*` | Capture + primitive buses to live reasoning context. |
| `DigitalCaptureProbe.*` | Deterministic contracts and live capture verification. |

## Capture contract

Successful frames are top-down `CV_8UC3` OpenCV matrices in `BGR8_SRGB`.
`DigitalCaptureMetadata::source_rect` uses physical virtual-desktop pixels and
may have a negative origin. The application opts into per-monitor-v2 DPI
awareness before any window is created.

Every attempt carries timestamps, backend and duration, source rectangle and
display DPI, foreground metadata, explicit status/error, and source device ID,
platform, and transport. Device origin prevents host-native enrichment from
being applied accidentally to a remote frame.

`DigitalFrameBus` clones a successful image once at publish time. Consumers pin
the immutable packet and share its `cv::Mat` storage.

## Windows capture backend

Capture uses a checked GDI DIB-section backend with `CAPTUREBLT`. Every resource
allocation and `BitBlt` result is validated. Monitor enumeration comes directly
from `EnumDisplayMonitors` on every capture. The GRIM overlay is marked
`WDA_EXCLUDEFROMCAPTURE`. A future Windows Graphics Capture backend can implement
`DigitalCaptureSource` without changing downstream consumers.

## OCR is the portable baseline

OCR consumes captured pixels and therefore works for any device whose frames can
enter `DigitalFrameBus`: local Windows/macOS/Linux capture, a remote iOS stream,
or a future browser/device agent. The default provider is Tesseract and can be
replaced behind `DigitalOcrProvider` without changing the bus, panel, or router.

OCR returns full text, line rectangles, confidence, provider status, timing, and
the exact source frame counter. A slow pass may skip intermediate frames, but it
never recaptures or labels results with the wrong frame.

## Optional native automation branch

For a locally captured native Windows frame, `DigitalAutomationProvider` queries
the foreground Windows UI Automation tree. It returns visible roles, names,
automation IDs, enabled state, and desktop bounds. Password controls are marked
and names are replaced with `[protected]`. The result records whether the target
changed between capture and inspection.

Native automation is preferred for structural grounding only when it succeeded,
returned elements, and still matches the captured foreground target. OCR remains
present in the same snapshot as the portable baseline and fallback text channel.
The loop will not run host Windows automation against remote, non-Windows, or
non-native frames. Other platforms can add optional providers later; none is a
dependency of the portable pipeline.

## Digital Environment panel

Open the panel with the `Digital` button in the Console toolbar. Capture controls
select the source and cadence, request a frame, pause/resume, refresh displays,
and toggle layered-window capture.

The `Capture`, `OCR`, and `Windows Automation` views show capture health or the
corresponding primitives. OCR regions use frame-local coordinates. Automation
controls map physical desktop rectangles into the captured source rectangle. A
frame-sync indicator prevents stale boxes from appearing to belong to a newer
preview.

## Probe

The executable contains an early-exit capture probe:

```powershell
GRIM.exe --digital-capture-probe --duration 60 --interval-ms 1000 `
  --mode active-monitor --output data/digital_capture_probe.png
```

Modes are `active-monitor`, `monitor`, `active-window`, and `virtual-desktop`.
Use `--monitor-index N` for a selected monitor.

## Primitive exit criteria

- Every successful frame produces OCR or an explicit unavailable, disabled, or
  failed status.
- Windows UI Automation produces typed controls for a matching local foreground
  window and fails explicitly elsewhere.
- Remote/non-Windows provenance cannot be mistaken for a local Windows target.
- OCR and automation share one immutable snapshot and source frame ID.
- Router context exposes OCR text/regions, UI elements, provider health, target
  correlation, device origin, and capture/primitive provenance.
- The panel visualizes exact-frame OCR/UI boxes and reports analysis lag.
- Neither primitive provider captures the screen or controls user input.

## Next phase boundary

The next semantic layer should merge overlapping OCR regions and native controls
into device-neutral grounded UI entities, then add on-demand visual-language
description for canvas, games, images, and video. Durable-memory/redaction policy
must be decided before any captured text is persisted.
