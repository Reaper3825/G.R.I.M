# Stage 1 — Physical Environment (Camera Input) Foundation

> **Status:** Complete. This document captures the foundational work, the
> component layout, and the non-obvious landmines discovered while wiring it up.
> Future stages (object detection, scene classification, SLAM) consume the
> `PhysicalFrameBus` defined here — treat this as the stable contract.

---

## 1. Goal

Deliver stable image/video input into the GRIM runtime from:

1. **Local device cameras** (macOS FaceTime camera, Windows webcams, Linux v4l2
   devices) addressed by integer index.
2. **Remote IP cameras** — both devices on the local NICs and devices
   registered with the GRIM hub (RTSP / HTTP / MJPEG URLs typed by the user).

Both paths share a single capture worker, a single published frame bus, and a
single UI surface.

---

## 2. Component Map

| Layer | File | Responsibility |
|---|---|---|
| Enumeration | `perception/physical/PhysicalCameraDirectory.cpp` | Scan local NICs, local device indices, and hub-registered devices. Produce a list of candidates with explicit `status_reason` for Disabled entries. |
| Platform NIC scan | `perception/physical/platform/scan_nics_macos.mm` / `_win32.cpp` / `_linux.cpp` | OS-specific IPv4 address enumeration. |
| Capture worker | `perception/physical/PhysicalCameraStream.cpp` | One worker thread per active source. Owns a `cv::VideoCapture`. Publishes `latest_frame_` under a mutex. |
| Frame bus | `perception/physical/PhysicalFrameBus.cpp` | Single global publish point. `PublishPhysicalFrameToBus()` + `PullLatestFrameView()`. Monotonic counter. |
| Tick orchestrator | `perception/physical/PhysicalEnvironmentLoop.cpp` | `TickPhysicalEnvironment()` called once per main-loop iteration. Drains the active stream into the bus. |
| UI panel | `ui/ui_physical_environment_panel.cpp` | Refresh / Source dropdown / URL field / Connect / Disconnect / live preview / status / error. |
| Entry wiring | `main.cpp` | `RegisterPhysicalEnvironmentDeviceServer()`, construct panel, add to `UIRoot`, call `TickPhysicalEnvironment()` in mainloop, call `ShutdownPhysicalEnvironment()` on teardown. |
| Console button | `ui/console_panel.cpp` | "Camera" button in the toolbox that calls `setVisible(true)` on the panel. |

---

## 3. URL Scheme

The `source_url` string is the universal identifier inside the subsystem.

| Scheme | Meaning | Open path |
|---|---|---|
| `device:N` | Local device index N | `cap.open(N, <platform backend list> → CAP_ANY)` |
| `rtsp://…` / `http://…` | Network stream | `cap.open(url, CAP_FFMPEG)` then fall back to `CAP_ANY` |

**Candidates with no `url_template`** are flagged `Disabled` and MUST carry a
populated `status_reason` explaining why (Rule 20 — fail loud, no silent
fallbacks). The UI shows `status_reason` as amber text beneath the URL field.

---

## 4. Enumeration Order (`RefreshPhysicalCameraDirectory`)

1. **LocalDevice** (indices 0..3) — probed with `cv::VideoCapture::open(i, CAP_*)`
   via the platform's native backend first, then `CAP_ANY`.
2. **LocalNic** — every IPv4 interface emits a `Disabled` candidate with
   `status_reason = "no URL template; type one manually"`. We never auto-dial
   our own IPs (see "Self-dial trap" below).
3. **HubDevice** — devices reported by `DeviceCommServer`.

---

## 5. Platform Backend Lists

Selected by `#if defined(__APPLE__) / _WIN32 / __linux__` in
`EnumerateLocalDeviceCameras`:

- **macOS:** `CAP_AVFOUNDATION` → `CAP_ANY`
- **Windows:** `CAP_DSHOW` → `CAP_MSMF` → `CAP_ANY`
- **Linux:** `CAP_V4L2` → `CAP_ANY`

On macOS, index-0 failure also includes TCC (camera permission) guidance in
`status_reason` so the user knows to check Settings → Privacy & Security.

---

## 6. Rendering Pipeline (Preview Path)

```
cv::VideoCapture (backend)
    ↓  cap.read(cv::Mat)             [worker thread]
PhysicalCameraStream::latest_frame_
    ↓  PullLatestFrameInto()         [main thread, inside Tick]
PhysicalEnvironmentLoop::pull_scratch
    ↓  PublishPhysicalFrameToBus()
PhysicalFrameBus::latest_image_
    ↓  PullLatestFrameView()         [UI panel]
UIPhysicalEnvironmentPanel::last_view_.image
    ↓  cv::resize() → manual per-pixel blit
OverlayRenderer pixel buffer (ARGB 32-bit)
    ↓  grimOverlayBlit()             [core/platform_window_macos.mm]
CGBitmapContext → CGImage → CALayer.contents → NSView
```

Two distinct ARGB conventions exist in this pipeline:

- **UI chrome** (buttons, text, rects): `(a<<24)|(r<<16)|(g<<8)|b` — the
  `OverlayRenderer::drawRect` convention.
- **Camera frames:** Empirically requires a non-standard pack. See the
  color-order landmine below.

---

## 7. ⚠️ The Color-Order Landmine

### Symptom
Yellow objects in the real world render as magenta in the preview.

### What the docs say
OpenCV `cv::VideoCapture::read` returns `CV_8UC3` in **BGR** byte order.
`OverlayRenderer::drawRect` packs ARGB as `0xAARRGGBB`.

### What actually happens on macOS arm64 + OpenCV 4.12 (vcpkg) + AVFoundation (CAP=1200)
Naming the bytes `b, g, r` by BGR convention and packing them `(A<<24)|(r<<16)|(g<<8)|b` produces a **G↔B inversion**. Yellow becomes magenta.

### Empirical proof
Diagnostic log (see §8) on an average warm-lit room:

```
FRAME DIAG frame#300 CV_8UC3 1920x1080 mean[c0,c1,c2]=[119,136,148] center_px=[55,75,97]
```

Mean(c2) > Mean(c1) > Mean(c0) — so c2 pairs with R and the order "looks" BGR
by aggregate statistics. But the visual result is unambiguous: with the
documented pack, yellow → magenta. The fix is to swap the B and G slots in
the output pack (not the source labels):

```cpp
// in ui_physical_environment_panel.cpp, DrawLatestFrameIntoOverlay
const uint8_t b = src[sx * 3 + 0];
const uint8_t g = src[sx * 3 + 1];
const uint8_t r = src[sx * 3 + 2];
dst[x] = (0xFFu << 24)
       | (static_cast<uint32_t>(r) << 16)
       | (static_cast<uint32_t>(b) << 8)   // <-- B in the G slot
       |  static_cast<uint32_t>(g);         // <-- G in the B slot
```

### Why this exists
The macOS display chain does a second R↔B swap in `grimOverlayBlit`
(see `core/platform_window_macos.mm` around lines 928–938) so that Core
Graphics' `PremultipliedLast` gets `[R, G, B, A]`. That swap is correct for
UI chrome because the chrome producer wrote `(A<<24)|(R<<16)|(G<<8)|B` —
equivalent to `[B, G, R, A]` in little-endian memory.

The camera-blit case composes differently: the display R↔B swap interacts
with the (actual) byte order AVFoundation delivers, and the net effect is
that G and B end up transposed in the visible image. The simplest,
platform-local, non-regression fix is the pack-slot swap above. Doing it
this way avoids modifying the shared `grimOverlayBlit` which works correctly
for all other panel rendering.

### When to revisit
Any of the following MUST trigger a revisit and re-verification with a
known-yellow subject:

- OpenCV major version change (4 → 5)
- vcpkg rebuild with different AVFoundation integration options
- macOS major version change (SDK change can alter CVPixelBuffer layout)
- Migrating off `grimOverlayBlit` (e.g., direct Metal texture upload)
- Adding Windows or Linux preview paths (they'll have their own color order;
  don't assume this pack works there)

---

## 8. Diagnostic Hook (Currently Disabled)

`PhysicalEnvironmentLoop.cpp::DrainActiveStreamLocked` has a disabled periodic
diagnostic that can be re-enabled by restoring the commented block. It logs:

```
FRAME DIAG frame#<N> CV_8UC<C> <cols>x<rows> mean[c0,c1,c2]=[..] center_px=[..]
```

Use this whenever you suspect the camera channel order changed under you.
Wait ~10s after connect for auto-exposure to settle before trusting the
numbers. Near-black frames (means near [10,10,10]) cannot reveal channel
order — you need a colorful, well-lit scene.

---

## 9. Self-Dial Trap (Fixed)

Early versions auto-populated `url_template` for every local NIC IPv4 as
`rtsp://<our-own-ip>:554/`. Connect attempts then hung trying to RTSP-dial
ourselves. Fix: LocalNic candidates emit **empty** `url_template` and are
marked `Disabled`. The UI requires the user to type an explicit URL.

---

## 10. Error-Log Dedup (Fixed)

`DrainActiveStreamLocked` stores `last_error_reason` with a
`"active stream failed: "` prefix. Its dedup check MUST compare against the
prefixed form. Comparing against the raw input string makes the check always
false and spams the log every tick.

---

## 11. Rule 20 Compliance

- No fallback URLs are auto-generated. Every Disabled candidate carries
  explicit `status_reason`.
- Empty-frame pulls in the worker are **fatal** — the stream transitions to
  `Failed` and the owner decides whether to reopen.
- `PublishPhysicalFrameToBus` throws on empty `cv::Mat`.
- `DrawLatestFrameIntoOverlay` fails loud (visible error text in panel) if
  `OverlayRenderer::getPixels()` returns null.

---

## 12. Consumption Contract

Downstream stages pull via:

```cpp
GRIM::Perception::Physical::PhysicalFrameBus::FrameView view;
uint64_t seen = 0;
if (PhysicalFrameBus::Instance().PullLatestFrameView(view, seen)) {
    // view.image is CV_8UC3 in OpenCV-backend-native byte order.
    // IMPORTANT: consumers doing inference should cvtColor to the exact order
    // their model expects. Do NOT assume BGR on macOS — see §7.
    // view.frame_counter, view.published_at, view.source_url, view.source_label
}
```

The bus is global, thread-safe, and one-shot per consumer (each consumer
maintains its own `last_seen_counter`).

---

## 13. Build Notes

- `cmake/Dependencies.cmake` and `cmake/Config.cmake` include `videoio` and
  `imgcodecs` in the `find_package(OpenCV … COMPONENTS …)` list. Omitting
  them produces a silent CMake success and a link-time `cv::VideoCapture`
  failure.
- Platform scan sources are picked up by `cmake/Sources.cmake`'s recursive
  glob over `perception/physical/` and then filtered by OS at the C++ level
  via `#if defined(...)` — there is no per-platform CMake list to maintain.

---

## 14. Known Non-Issues

- **Warm/pink indoor tint:** That's real-world lighting plus the MacBook
  camera's auto-white-balance. Not a pipeline bug.
- **Status=STREAMING with 29.9 fps instead of 30:** Normal — `measured_fps_`
  is computed over 1-second windows, jitter of ±0.1 is expected.
- **Local device 0 "not present" on first macOS run:** TCC permission
  dialog. Accept, relaunch GRIM.

---

## 15. Future-Proofing Checklist

When adding a new capture backend or platform:

- [ ] Add backend IDs to `EnumerateLocalDeviceCameras` platform list.
- [ ] Re-enable the FRAME DIAG block and verify channel order with a yellow
      subject. Update the color-order landmine section if the pack must
      change for the new platform.
- [ ] If the new backend delivers non-CV_8UC3 frames (e.g., YUV, RAW16),
      add a `cv::cvtColor` normalization step in the worker or publish path
      and document it here.
- [ ] Write the `status_reason` for every Disabled case — no silent skips.
