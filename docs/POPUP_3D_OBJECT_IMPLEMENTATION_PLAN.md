# Popup 3D Object — Implementation Plan

> Replaces the current popup sprite pipeline with a **single, simple, native 3D object renderer**.
>
> This is a **cutover plan**, not a compatibility plan. The old sprite path is removed rather than supported in parallel.

---

## Locked Design Decisions

These decisions are fixed for implementation unless a later explicit design review changes them.

| # | Decision |
|---|----------|
| 1 | **No backwards compatibility.** The popup becomes a 3D object renderer only. No dual sprite/3D mode. |
| 2 | **Single object path.** The system is built for one popup object at a time, not a general scene engine. |
| 3 | **Transparent background remains mandatory.** The visible popup uses a transparent platform window/surface with per-pixel alpha semantics. |
| 4 | **bgfx renders offscreen only.** The final visible popup is presented through a platform-specific CPU-frame presenter, not direct bgfx presentation to the popup surface. |
| 5 | **Project-native logic only.** Use existing project dependencies (`bgfx`, `bx`, `bimg`, `stb`) and local code only. No Assimp, no external scene system, no generic model framework. |
| 6 | **Simple material model.** One albedo texture + one packed material texture. No full PBR, no shadow maps, no skeletons, no scene graph. |
| 7 | **Modular file layout.** Rendering core, material handling, mesh handling, object definition, and window presentation live in separate files to avoid bloated includes and mixed responsibilities. |
| 8 | **Plain-data architecture over inheritance.** Prefer small structs and explicit builder functions over deep abstract class hierarchies. |
| 9 | **Cross-machine paths only.** No hardcoded `D:/...` resource paths remain in the popup pipeline. |
| 10 | **Windows and macOS Metal are both supported targets.** The renderer core is shared; popup presentation is platform-specific (`UpdateLayeredWindow` on Windows, Cocoa/CALayer blit on macOS). |
| 11 | **A single global bgfx API thread owns submission for the entire executable.** If bgfx is moved off the UI/main thread, that dedicated thread becomes the one global bgfx API thread for the whole process, not a popup-local second bgfx runtime. |
| 12 | **Readback is asynchronous and delayed.** The renderer never blocks waiting for the frame it just submitted. |
| 13 | **Readback staging is explicit.** The popup renderer uses a 3-slot readback ring plus one thread-safe mailbox copy for handoff to the presenter. |
| 14 | **Popup offscreen format is locked for v1.** Color = `BGRA8`, depth = `D24S8`, single-sample only, `BGFX_TEXTURE_READ_BACK` enabled. |
| 15 | **Premultiplication happens in the presenter.** Shaders and renderer produce straight-alpha color; `popup_window.cpp` converts to the platform-required CPU presentation format immediately before final presentation. |
| 16 | **`WindowManager` remains the sole global bgfx owner.** It owns init/reset/shutdown/platform-data/frame pumping. The popup renderer owns only popup-scoped GPU resources. |
| 17 | **Coordinate convention is fixed.** Right-handed world, `+X` right, `+Y` up, `+Z` toward camera, counter-clockwise front faces, `BGFX_STATE_CULL_CW`. |
| 18 | **No per-frame GPU rebuilds.** Meshes, textures, shader programs, and static uniforms are created once and reused until shutdown or explicit resize/reset recreation. |
| 19 | **Popup close/hide uses drain-then-destroy.** New popup submissions stop immediately; pending readbacks are drained before popup GPU resources are destroyed; unread completed frames are discarded. |
| 20 | **Mailbox exchange is mutex-protected copy + generation counter.** No lock-free pointer aliasing or shared mutable buffer between renderer and presenter. |
| 21 | **Uniform handles and uniform values have different lifetimes.** Handles are created once and destroyed at renderer teardown; values are uploaded at submit time only. |
| 22 | **Popup renderer failures are fail-hard.** Missing shader binaries, missing texture assets, framebuffer creation failure, or readback invariant violations stop execution loudly rather than silently degrading. |
| 23 | **macOS AppKit/Cocoa window mutations stay on the AppKit thread.** Only bgfx API submission moves to the dedicated submission thread. |
| 24 | **Exactly one bgfx instance exists per executable.** No popup-local `bgfx::init`, no second `WindowManager`, no second platform-data owner, and no second independent bgfx frame loop. |

---

## Problem Statement

The current popup path is not a real 3D renderer:

- `popup_ui/popup_window.cpp` creates a transparent layered window and performs CPU-side pixel composition.
- `popup_ui/popup_renderer.cpp` loads images on the CPU only.
- `resources/shaders/vs_sprite.sc` and `resources/shaders/fs_sprite.sc` are only sufficient for a textured 2D quad with alpha.
- Lighting is currently simulated by CPU-side mask blending rather than actual normals, camera transforms, or light equations.

That architecture cannot cleanly support a true textured 3D object with native lighting while preserving a transparent desktop background.

The correct architecture is:

1. Render the 3D object offscreen with bgfx into a straight-alpha `BGRA8` target cleared to transparent.
2. Read back the rendered pixels.
3. Convert to the platform-required CPU presentation format and present through the platform presenter.

This keeps the background fully transparent while allowing native 3D shading.

---

## Goals

1. Replace the sprite popup with a single 3D popup object renderer.
2. Keep the desktop background fully transparent outside the object silhouette.
3. Support real lighting using mesh normals and shader-based shading.
4. Keep the implementation small, explicit, and modular.
5. Avoid generic rendering abstractions that would overcomplicate a one-object popup.
6. Remove the old sprite code instead of layering new logic on top of it.

---

## Non-Goals

These are intentionally excluded from the first implementation:

- full PBR workflow
- shadow mapping
- skeletal animation
- model import framework with many file formats
- generic scene graph
- fallback sprite rendering path
- old shader compatibility layer
- multi-object popup scene management

---

## Final Architecture

### High-level data flow

```text
Platform presenter/UI thread
	Windows: popup UI thread
	macOS: AppKit/main thread
	pop_ui.cpp (or platform popup UI host)
		-> updates PopupAnimState
		-> writes latest PopupRenderInput snapshot
	PopupWindowPresenter (popup_window.cpp)
		-> reads PopupFrameMailbox
		-> converts straight-alpha BGRA8 to premultiplied BGRA8 when needed
		-> presents via platform-specific path

Dedicated submission thread (single global bgfx API thread)
	WindowManager submission-thread loop
		-> owns the process-wide bgfx API submission path on the submission thread
		-> pumps one bgfx frame while popup is visible or readback is pending
		-> calls Popup3DRenderer::submitAndPoll()
	Popup3DRenderer
		-> reuses prebuilt mesh/material/object state
		-> renders offscreen via bgfx into one staging slot
		-> queues asynchronous texture readback
		-> publishes newest completed raw BGRA8 frame to PopupFrameMailbox
```

### Responsibility split

| Area | Responsibility |
|------|----------------|
| `pop_ui.cpp` | Popup loop, animation state updates, show/hide timing, render orchestration |
| `WindowManager` | Sole global bgfx owner: init/reset/shutdown/platform data/submission-thread lifetime/frame pump/reset epoch |
| popup window presenter | Platform-specific window creation and pixel presentation only |
| 3D renderer | bgfx resources, view/projection, offscreen render pass, readback |
| frame mailbox | Mutex-protected full-frame raw-buffer copy plus generation counter handoff from the submission thread to the presenter/UI thread |
| mesh module | Vertex/index buffer data and GPU upload |
| material module | Texture load, uniform creation, shader binding |
| object definition file | The actual popup object definition (geometry/material defaults/transform defaults) |

---

## Target File Layout

The new layout should stay intentionally small.

```text
popup_ui/
	popup_window.hpp/.cpp                # platform-specific popup presenter (Win32 layered window / macOS Cocoa blit)
	popup_3d_types.hpp                  # tiny POD structs/enums only
	popup_3d_mesh.hpp/.cpp              # mesh CPU data + bgfx buffer ownership
	popup_3d_material.hpp/.cpp          # textures, uniforms, shader binding
	popup_3d_shaders.hpp/.cpp           # program load/create + shader uniform handles
	popup_3d_renderer.hpp/.cpp          # offscreen bgfx render + readback
	popup_3d_mailbox.hpp/.cpp           # mutex-protected raw-frame handoff + generation counter
	popup_3d_assets.hpp/.cpp            # cross-platform resource path resolution
	objects/
		grim_popup_object.hpp/.cpp        # concrete popup object definition

resources/shaders/
	vs_popup_model.sc
	fs_popup_model.sc
	common_popup_model.sh               # shared varying/uniform declarations if needed

resources/popup_3d/
	grim_popup_albedo.png
	grim_popup_packed.png               # R=occlusion, G=roughness, B=emissive, A=opacity

docs/
	POPUP_3D_OBJECT_IMPLEMENTATION_PLAN.md
```

### Files explicitly removed or retired

The cutover is cleaner if these are deleted or reduced to non-compiled references:

- `popup_ui/popup_renderer.hpp`
- `popup_ui/popup_renderer.cpp`
- `resources/shaders/vs_sprite.sc`
- `resources/shaders/fs_sprite.sc`
- `resources/shaders/common_sprite.sh`

If temporary reference retention is needed during development, rename to `_OLD` and ensure they are not part of the build or live code path.

---

## Minimal Data Model

### `popup_3d_types.hpp`

This header must stay small and dependency-light.

It should contain only plain structs such as:

- `PopupVertex`
- `PopupTransform`
- `PopupLightParams`
- `PopupRenderInput`
- `PopupFrameBuffer`
- `PopupReadbackSlot`
- `PopupFrameMailbox`
- `PopupObjectDefinition`
- `PopupRenderFrame`

### Why this matters

This keeps most headers from including heavy bgfx headers everywhere.

Rule for includes:

- headers expose declarations and POD structs only
- `.cpp` files include `bgfx/bgfx.h`, `bx/math.h`, stb headers, and other heavy dependencies
- object-definition headers include only `popup_3d_types.hpp`

This avoids include sprawl and keeps compile cost sane.

---

## Thread and Lifetime Ownership Contract

This section is authoritative. If code and prose disagree, follow this section.

### Thread model

- **Single global bgfx API thread** owns all bgfx API calls for the entire executable.
- If bgfx is moved off the UI/main thread, that thread is the dedicated submission thread.
- The popup renderer is a client of that one global bgfx instance; it does not get its own bgfx lifecycle.
- **Presenter/UI thread** owns popup visibility, timing, and CPU-frame presentation only.
- `pop_ui.cpp` does **not** submit draw calls, call `bgfx::frame()`, or create/destroy popup GPU resources.
- On **macOS**, `NSWindow` / `NSView` / `CALayer` creation and mutation remain on the AppKit/main thread.

This is a target-state change from the current repo architecture.

- today, bgfx is initialized and pumped from the main thread
- if the app adopts a dedicated submission thread, that thread becomes the single global bgfx API thread for the whole process
- platform window creation stays on the OS-required UI thread

### Single bgfx instance rule

The process owns exactly one bgfx runtime:

- one `bgfx::init`
- one `bgfx::shutdown`
- one `bgfx::setPlatformData` owner
- one `bgfx::frame` pump loop

Popup 3D rendering must plug into that existing global runtime.

Forbidden:

- popup-local `bgfx::init`
- popup-local `bgfx::shutdown`
- second `WindowManager`
- second independent bgfx frame loop
- second platform-data owner

### bgfx threading mode rule

- Use a dedicated **process-wide bgfx API/submission thread** owned by `WindowManager`.
- Do **not** call `bgfx::renderFrame()` before `bgfx::init()` as part of this design; that would force a different bgfx threading mode than this plan intends.
- That submission thread uses the normal `bgfx::init()` + `bgfx::frame()` path.

### Global ownership map

| Owner | Owns |
|------|------|
| `WindowManager` | the single global `bgfx::init`, `bgfx::reset`, `bgfx::shutdown`, `bgfx::frame`, `bgfx::setPlatformData`, renderer reset epoch, and the dedicated submission-thread pump |
| `Popup3DRenderer` | Popup view ID, popup shader program, immutable uniform handles, popup mesh/material GPU resources, offscreen framebuffer(s), readback ring, and mailbox publication |
| `PopupWindowPresenter` (`popup_window.cpp`) | Platform popup window handle, presenter scratch objects/buffers, and OS-specific CPU-frame presentation |
| `pop_ui.cpp` | Animation state, visibility state, and the latest `PopupRenderInput` snapshot |
| object-definition files | Immutable CPU-side object description only |

### View ID contract

- Reserve a dedicated popup offscreen view ID: **`31`**.
- No other subsystem may submit popup geometry through any other view.
- `Popup3DRenderer` is the sole owner of view `31`.

### Reset / resize contract

- `WindowManager` remains the only code allowed to call `bgfx::reset`.
- Add a monotonically increasing **reset epoch** to `WindowManager`.
- `Popup3DRenderer` stores `lastSeenResetEpoch` and recreates size-dependent popup render targets whenever:
	- the popup render size changes, or
	- the reset epoch changes.
- Static popup GPU resources (mesh, textures, shader program, static uniforms) are **not** rebuilt per frame.

### Frame-pump contract

The current hidden-window bgfx keepalive throttles to every 30th frame. That is not valid for popup readback.

When the 3D popup renderer is active:

- the dedicated submission-thread loop must advance **one bgfx frame every render tick** while:
	- the popup is visible, or
	- any popup readback slot is still pending.

No 2-fps keepalive throttling is allowed in that state.

This is still the **one global bgfx frame loop** for the executable, not a popup-local second loop.

### Close / hide / shutdown drain contract

Popup close/hide does **not** immediately destroy popup-scoped GPU resources.

The rule is:

1. stop enqueuing new popup draws and new popup readbacks immediately
2. continue dedicated submission-thread bgfx frame pumping in **drain mode**
3. keep polling until every readback slot is no longer `PendingReadback`
4. discard any unread `Ready` slot contents and discard the mailbox contents
5. destroy popup-scoped GPU resources

Because the readback ring size is fixed at 3 and no new readbacks are queued during drain mode, draining is bounded to at most **3 additional bgfx frame advances**.

Application shutdown follows the same rule:

- popup renderer drain completes first
- popup-scoped GPU resources are destroyed second
- `WindowManager::shutdown()` may proceed last

There is no “destroy immediately and hope pending readbacks don’t matter” path in v1.

---

## Object Definition Strategy

The object system should be **modular but not abstract for abstraction’s sake**.

### Required rule

Each popup object lives in its own file pair under `popup_ui/objects/`.

For the first implementation there is only one object:

- `popup_ui/objects/grim_popup_object.hpp`
- `popup_ui/objects/grim_popup_object.cpp`

### What the object file owns

The object definition file should describe:

- base transform defaults
- mesh construction for the popup object
- material asset paths
- rotation/pivot defaults
- optional animation tuning defaults

### What it does **not** own

It should **not** own:

- bgfx init/shutdown
- platform popup presentation
- shader compilation
- general popup loop timing

### Recommended shape

Use builder functions, not an inheritance tree.

Example conceptually:

- `PopupObjectDefinition createGrimPopupObject(const PopupAssetPaths& assets);`

That is enough. No interface class is required for version one.

If a second object is added later, add a second file pair and a small explicit factory file. Do not pre-build a generic plugin-style object registry.

---

## Mesh Plan

### Vertex format

The first 3D path needs:

- position: `vec3`
- normal: `vec3`
- uv: `vec2`

This is the minimum viable format for textured directional lighting.

### Mesh source

The first implementation should not depend on a model import library.

Recommended rollout:

1. **Phase 1 mesh**: hardcoded test mesh or hardcoded final popup mesh data
2. **Optional later**: tiny project-native OBJ loader if hand-authored external mesh files become necessary

### Ownership

`popup_3d_mesh.cpp` owns:

- CPU mesh arrays
- bgfx vertex layout creation
- vertex buffer creation
- index buffer creation
- GPU handle destruction

---

## Material Plan

The popup should use one simple material definition:

- albedo texture
- packed material texture

Packed texture channels:

- `R = occlusion`
- `G = roughness`
- `B = emissive`
- `A = opacity`

### Why keep this format

This matches the spirit of the current popup asset flow while moving the actual lighting into shaders.

### `popup_3d_material.cpp` owns

- loading texture data
- creating bgfx textures
- creating sampler uniforms
- binding textures for a draw call
- destroying texture and uniform handles

### Uniform handle/value lifetime contract

There are two distinct categories and they must not be mixed:

#### Immutable handles (created once)

Created during renderer/material/shader initialization and destroyed during popup renderer teardown:

- sampler uniform handles
- non-sampler uniform handles (alpha, emissive, light direction/intensity, material params)
- shader program handles

These are long-lived bgfx handles, not per-frame data.

#### Per-frame values (uploaded at submit time)

Updated every submit as plain POD values:

- model matrix
- view/projection matrices
- alpha multiplier
- emissive multiplier
- light direction/intensity values
- any per-frame material scalar values

These are uploaded with `bgfx::setTransform` / `bgfx::setUniform` during submission and do not own bgfx lifetime.

`Popup3DRenderer` owns the immutable uniform handles.
Object-definition files never own bgfx handles.

---

## Shader Plan

The current sprite shaders are not sufficient.

### New shader set

- `resources/shaders/vs_popup_model.sc`
- `resources/shaders/fs_popup_model.sc`
- `resources/shaders/common_popup_model.sh` (only if shared declarations make it cleaner)

### Vertex shader responsibilities

- transform position by model/view/projection
- transform normal into world or view space
- pass UV to fragment shader
- pass lighting inputs needed by the fragment shader

### Fragment shader responsibilities

- sample albedo texture
- sample packed material texture
- read opacity from packed alpha
- compute ambient + directional light
- optionally add roughness-driven highlight approximation
- optionally add emissive boost
- write final straight-alpha color with preserved alpha

### Lighting model

First implementation uses:

- ambient term
- one directional light
- simple specular approximation

Do **not** implement full PBR or shadow maps in the first cut.

### Color-space contract

- Albedo texture content is treated as artist-authored sRGB image data.
- Lighting is computed in **linear** space.
- Final RGB is converted back to display/gamma space in the fragment shader before writing to the offscreen color target.
- Alpha is never gamma-corrected; it remains a straight coverage/opacity value.
- The offscreen target itself is a plain `BGRA8` UNORM target for deterministic readback and deterministic presenter conversion.

---

## Offscreen Render Plan

`popup_3d_renderer.cpp` is the core of the cutover.

### Renderer responsibilities

It owns:

- offscreen color target creation
- depth target creation
- framebuffer creation
- view rect setup
- view/projection matrices
- model transform submission
- draw-state setup
- bgfx frame submission
- readback into CPU-visible raw BGRA buffer

### Required render target properties

- color format = `bgfx::TextureFormat::BGRA8`
- texture flags include `BGFX_TEXTURE_RT | BGFX_TEXTURE_READ_BACK`
- depth format = `bgfx::TextureFormat::D24S8`
- single-sample only in v1 (no MSAA on the popup readback path)
- clear color must be transparent: `BGRA = (0, 0, 0, 0)`
- renderer initialization must fail loudly if `BGFX_CAPS_TEXTURE_READ_BACK` is unavailable

### Required bgfx state

- RGB write
- alpha write
- depth write
- depth test
- cull back faces
- no MSAA in v1
- alpha blending enabled only if the object material actually requires it

### Important rule

The renderer must never paint a full opaque background. If it clears to opaque, desktop transparency is gone.

### Readback contract

The readback path is **explicitly asynchronous**.

Use bgfx semantics directly:

- `bgfx::readTexture(...)` returns a **frame number when the result will be available**.
- `bgfx::frame()` advances the frame pump and yields the current frame number.

The popup renderer must never spin, wait, or block for the frame it just submitted.

#### Required staging objects

`Popup3DRenderer` owns:

- `PopupReadbackSlot slots[3]`
- one `PopupFrameMailbox latestCompletedFrame`

Each readback slot contains:

- one color texture (`BGRA8`, readback enabled)
- one depth texture (`D24S8`)
- one framebuffer handle
- one raw **straight-alpha** CPU buffer sized `width * height * 4`
- one `readyAfterFrame` value returned from `bgfx::readTexture`
- one generation counter
- one state enum: `Idle | PendingReadback | Ready`

#### Submission and ownership sequence

For each dedicated submission-thread popup render tick:

1. Poll existing slots.
2. If `currentBgfxFrame >= slot.readyAfterFrame`, that slot becomes `Ready`.
3. Copy the newest ready slot into `PopupFrameMailbox` and increment mailbox generation.
4. Choose the next `Idle` slot for rendering.
5. If no slot is idle, **skip submission** and keep the last completed frame; do not stall.
6. Render the popup object into that slot's framebuffer.
7. Queue `bgfx::readTexture(slot.colorTexture, slot.rawStraightBgra.data())`.
8. Record the returned `readyAfterFrame`.
9. Mark the slot `PendingReadback`.

#### Immediate vs delayed behavior

- A frame submitted this tick is **not** eligible for presentation this tick.
- Presentation always uses the **newest completed mailbox frame**, never the just-submitted frame.
- If no new frame is ready, the presenter reuses the last premultiplied presentation buffer instead of blocking the app.

#### CPU buffer ownership

- Readback-slot raw buffers are owned exclusively by `Popup3DRenderer`.
- `PopupFrameMailbox` owns one thread-safe copy of the newest completed **straight-alpha BGRA8** frame.
- `PopupWindowPresenter` owns a separate **premultiplied BGRA8** presentation scratch buffer.
- The presenter never mutates renderer-owned readback-slot buffers.

#### Mailbox exchange model

`PopupFrameMailbox` uses a **mutex-protected full-frame copy plus generation counter**.

It owns:

- one `std::vector<uint8_t>` straight-alpha raw `BGRA8` buffer
- width/height/stride metadata
- one monotonically increasing `generation`
- one `std::mutex`

Writer behavior (dedicated submission thread):

1. lock mailbox mutex
2. resize mailbox storage if dimensions changed
3. copy newest completed readback-slot buffer into mailbox-owned storage
4. update width/height/stride metadata
5. increment `generation`
6. unlock mutex

Reader behavior (presenter/UI thread):

1. lock mailbox mutex
2. compare `generation` with `lastConsumedGeneration`
3. if unchanged, unlock and reuse previous presenter buffers
4. if changed, copy mailbox raw buffer into presenter-owned raw scratch, store new generation, unlock mutex
5. perform premultiplication **after unlocking** into presenter-owned premultiplied scratch

This means:

- no zero-copy aliasing between threads
- no atomic pointer/index handoff model
- no possibility of the presenter reading a buffer while the renderer mutates that same storage

#### Premultiplication location

Premultiplication or equivalent platform-format conversion happens **only once**, in the presenter, on the mailbox copy immediately before final platform presentation.

That means:

- shader output = straight alpha
- mailbox buffer = straight-alpha `BGRA8`
- presenter scratch = premultiplied `BGRA8`

### MSAA / resolve rule

Version one intentionally forbids MSAA on the popup readback path.

If MSAA is ever added later, the contract becomes:

1. render into an MSAA target
2. resolve/blit into a **single-sample `BGRA8` readback texture**
3. call `bgfx::readTexture` only on the resolved single-sample texture

Readback from an MSAA target directly is not part of the v1 contract.

### Runtime invariant rule

If a readback completes with a size that does not match the slot dimensions or the current mailbox copy contract after resize/reset, that is an invariant violation.

In v1 this is **fail-hard**:

- log a detailed error
- stop popup rendering
- request main-loop stop
- do not continue with stale or partially sized buffers

---

## Camera and Mesh Convention Contract

This section fixes all default spatial conventions for the popup object path.

### Coordinate system

- handedness: **right-handed**
- `+X` = screen right
- `+Y` = screen up
- `+Z` = toward the camera

### Camera defaults

- camera position: `(0.0f, 0.0f, +2.5f)`
- camera target: `(0.0f, 0.0f, 0.0f)`
- up vector: `(0.0f, +1.0f, 0.0f)`
- projection: perspective
- vertical FOV: `30°`
- near plane: `0.05f`
- far plane: `10.0f`

### Mesh authoring rules

- the popup object is authored around the origin
- the local pivot is at the origin
- outward-facing triangles are **counter-clockwise** when viewed from outside the mesh
- the renderer uses `BGFX_STATE_CULL_CW`, leaving CCW faces visible
- normals point outward from the visible surface

### Animation / pivot rules

- uniform scale occurs around the local origin
- idle rotation and voice pulse operate on the object, not on the camera
- v1 does **not** orbit the camera; the object animates in place

### Lighting-space rule

- directional light is defined in world space
- normals are transformed from model space into world or view space in the vertex shader
- default light direction should be normalized and come from upper-front-left so the face visible to the user is lit by default

---

## Window Presentation Plan

`popup_window.cpp` needs to get much simpler.

### After cutover, `popup_window.cpp` should only do:

- create/manage the platform popup window handle
- own the platform-specific CPU-frame presentation code
- accept a prepared straight-alpha BGRA buffer from `PopupFrameMailbox`
- convert straight-alpha BGRA to the platform presentation format in presenter-owned scratch memory
- call the platform-specific CPU-frame presenter

### It should no longer do:

- texture loading
- CPU fake lighting
- roughness/emissive mask math
- occlusion-derived shadow generation
- sprite-specific asset loading

This file becomes a presenter, not a renderer.

### Presenter contract

- the presenter never touches bgfx objects
- the presenter never performs mesh/material/shader work
- the presenter may cache and reuse the last good premultiplied frame when no newer readback has completed
- the presenter is the only place where platform pixel-presentation rules are handled

### Platform presenter contract

#### Windows

- presenter thread = popup UI thread
- window type = Win32 layered popup window
- presentation path = premultiplied `BGRA8` + `UpdateLayeredWindow`

#### macOS

- presenter thread = AppKit/main thread
- window type = Cocoa popup/overlay window
- presentation path = CPU pixel blit into a Cocoa/CoreAnimation-backed surface (same family as the existing `grimOverlayBlit` path)
- `NSWindow` / `NSView` / `CALayer` mutations must never occur from the dedicated submission thread

---

## Animation Integration Plan

`pop_ui.cpp` already owns timing and popup animation state.

That should continue.

### Existing animation values to reuse

- scale
- alpha
- voiceIntensity

### New mapping into the 3D object path

- `scale` -> object scale multiplier and/or popup window presentation scale
- `alpha` -> final material alpha multiplier
- `voiceIntensity` -> emissive multiplier and subtle rotation/pulse amount

### Important simplification

The animation loop should not know mesh or shader details.

It should only prepare and publish a simple render input struct for the dedicated submission thread to consume.

### Cross-thread input rule

- `pop_ui.cpp` publishes a `PopupRenderInput` snapshot
- the single global bgfx API thread consumes the newest snapshot during the submission-thread loop
- no bgfx state is mutated directly from the presenter/UI thread

---

## One-Time vs Per-Frame Resource Contract

The original phrase “renderer builds mesh/material/object state” is too loose and must not be implemented literally.

### One-time creation only

These are created once during popup renderer initialization and reused:

- vertex layout
- mesh GPU buffers
- texture handles
- immutable uniform handles
- shader program
- immutable object definition data

### Recreated only on resize/reset

These are recreated only when popup dimensions or reset epoch change:

- offscreen color textures
- offscreen depth textures
- framebuffers
- readback-slot CPU buffers sized to the target
- presenter scratch buffer sized to the target

### Per-frame updates only

These are the only things allowed to change every frame:

- model matrix
- view/projection matrix if aspect ratio changed
- alpha multiplier
- emissive multiplier / voice-intensity uniforms
- light direction/intensity uniforms
- which readback slot is used for this frame

### Forbidden per-frame work

The render loop must **not** do any of the following every frame:

- recreate mesh buffers
- reload textures from disk
- recreate shader programs
- recreate static uniforms
- rebuild object definitions

If implementation does any of the above, it violates this plan.

---

## Asset Path Plan

All popup 3D assets must resolve from project-relative paths.

### New rule

No code in the popup renderer may contain hardcoded absolute drive paths.

### `popup_3d_assets.cpp` should own:

- repo-root-relative asset path construction
- validation that required assets exist
- a single returned `PopupAssetPaths` struct

This keeps asset policy in one place instead of scattering string literals across multiple files.

---

## Build / Packaging Plan

The project needs a real shader asset path, not the current mostly-stubbed arrangement.

### Required additions

1. Add a dedicated shader build/copy step for:
	 - `vs_popup_model.sc`
	 - `fs_popup_model.sc`
2. Ensure runtime output contains compiled shader binaries or embedded shader data.
3. Ensure `resources/popup_3d/` assets are copied into the runtime location.

### Recommended organization

If build logic becomes more than a few lines, move it into a dedicated CMake helper such as:

- `cmake/BgfxShaders.cmake`

That keeps `CMakeLists.txt` from becoming shader-tool glue soup.

---

## Failure Policy (Fail-Hard)

Popup 3D rendering follows GRIM fail-loud rules. There are no silent fallbacks, placeholder assets, or degraded legacy sprite paths.

### Fatal initialization failures

The following are fatal and must stop execution loudly:

- shader binary missing
- texture asset missing
- popup offscreen color texture creation failure
- popup depth texture creation failure
- popup framebuffer creation failure
- missing `BGFX_CAPS_TEXTURE_READ_BACK`

Required behavior:

- emit a detailed error message
- abort popup renderer initialization immediately
- if this occurs during startup, fail startup
- if this occurs after startup, request main-loop stop

### Fatal runtime invariant failures

The following are also fatal:

- readback size mismatch after resize/reset
- invalid mailbox metadata after copy
- unexpected invalid popup GPU handle in the live render path

Required behavior:

- emit a detailed error message with the expected vs actual values
- stop popup rendering immediately
- request main-loop stop
- do not attempt recovery by switching formats, skipping premultiplication, or falling back to the deleted sprite path

---

## Include Discipline Rules

To avoid bloated includes, implementation must follow these rules:

1. `popup_3d_types.hpp` contains POD structs only.
2. `popup_3d_mesh.hpp`, `popup_3d_material.hpp`, and `popup_3d_renderer.hpp` expose minimal public APIs.
3. bgfx-heavy headers are included in `.cpp` files wherever possible.
4. No object-definition header may include popup loop headers.
5. No window presenter header may include mesh or shader headers.
6. Do not dump all popup logic into one umbrella header.

If a header starts accumulating unrelated responsibilities, split it immediately.

---

## Cutover Plan (No Compatibility Layer)

This feature should land as a clean replacement, not a branch maze.

### Remove these legacy responsibilities

From `popup_window.cpp`:

- `queueWindowAlphaReadback()`
- CPU diffuse/oreo texture synthesis
- shadow-mask generation
- fake lighting composition

From the popup module overall:

- sprite-only renderer helpers
- sprite shader usage
- absolute sprite asset paths

### Resulting simplified flow

```text
old:
	load sprite textures -> derive CPU masks -> fake light on CPU -> platform popup presentation

new:
	render 3D object offscreen with bgfx -> read back straight-alpha BGRA8 -> platform popup presentation
```

---

## Implementation Phases

### Phase 1 — Scaffold the new modules

Create the new file structure with empty/minimal implementations:

- `popup_3d_types.hpp`
- `popup_3d_mesh.hpp/.cpp`
- `popup_3d_material.hpp/.cpp`
- `popup_3d_shaders.hpp/.cpp`
- `popup_3d_renderer.hpp/.cpp`
- `popup_3d_mailbox.hpp/.cpp`
- `popup_3d_assets.hpp/.cpp`
- `popup_ui/objects/grim_popup_object.hpp/.cpp`

Deliverable:

- project builds with new modules present
- no behavior change yet
- ownership boundaries are declared in code skeletons

### Phase 2 — Add shader and asset plumbing

Add:

- new shader source files
- build/package rules for shader runtime artifacts
- popup 3D asset copy rules

Deliverable:

- renderer can load shaders and textures successfully

### Phase 3 — Implement offscreen renderer

Add:

- vertex layout
- mesh upload
- material binding
- offscreen framebuffer
- camera/projection setup
- transparent clear
- 3-slot readback ring
- mailbox publication path
- `BGFX_CAPS_TEXTURE_READ_BACK` fail-loud validation

Deliverable:

- renderer can produce a delayed straight-alpha `BGRA8` frame with transparent background

### Phase 4 — Define the first popup object

In `grim_popup_object.cpp` implement:

- mesh creation or mesh loading
- material asset references
- transform defaults
- simple animation tuning defaults

Deliverable:

- one stable popup object renders correctly offscreen

### Phase 5 — Replace the popup presentation input

Refactor `popup_window.cpp` so it only presents provided straight-alpha `BGRA8` buffers.

Deliverable:

- platform popup presentation works from straight-alpha `BGRA8` mailbox input via presenter-side conversion/premultiplication

### Phase 6 — Integrate into `pop_ui.cpp`

Replace the sprite/CPU shading path with:

- animation update
- render-input snapshot publish
- present call

Deliverable:

- popup shows the 3D object instead of the sprite

### Phase 7 — Delete legacy sprite path

Remove or retire:

- sprite renderer files
- sprite shader files
- sprite-specific texture-loading code
- dead docs/comments that describe the old live path

Deliverable:

- one code path remains

---

## Validation Checklist

The cutover is not done until all of the following are true:

- [ ] popup still opens and closes correctly
- [ ] desktop remains visible through fully transparent background regions
- [ ] object is lit by shader-based lighting, not CPU tint math
- [ ] alpha edges are correct after premultiplication
- [ ] no hardcoded absolute asset paths remain in popup rendering code
- [ ] old sprite pipeline is no longer reachable
- [ ] popup animation still responds to `scale`, `alpha`, and `voiceIntensity`
- [ ] build includes shader assets and popup 3D textures correctly
- [ ] no new include bloat or monolithic popup header was introduced
- [ ] bgfx submission occurs on the single global bgfx API thread only
- [ ] popup rendering does not create a second bgfx instance, second `WindowManager`, or second independent bgfx frame loop
- [ ] no per-frame GPU resource rebuilds occur
- [ ] readback follows frame-number ownership rules and never blocks current-frame presentation
- [ ] popup offscreen format is `BGRA8 + D24S8`, single-sample, with no implicit MSAA readback
- [ ] camera conventions, winding, and culling produce a front-facing correctly lit object
- [ ] popup close/hide drains pending readbacks before popup GPU resource destruction
- [ ] mailbox exchange uses mutex-protected full-frame copy with generation counter, not shared mutable storage
- [ ] immutable uniform handles are created once while per-frame values upload only at submit time
- [ ] missing shaders/assets, framebuffer creation failure, and readback invariant violations fail hard with no fallback path
- [ ] macOS presenter uses Cocoa/CoreAnimation blit on the AppKit thread while bgfx stays on the dedicated submission thread

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Offscreen alpha comes back opaque | Explicitly clear render target alpha to 0 and validate readback before presentation |
| Readback stalls performance | Use delayed asynchronous readback, 3 staging slots, and reuse last completed frame instead of waiting |
| Frame ownership becomes ambiguous | Use slot generation IDs, mailbox generation IDs, and frame-number readiness from `bgfx::readTexture` |
| Shader asset loading fails | Add deterministic build/copy rules and startup validation with hard errors |
| Include sprawl returns | Keep `popup_3d_types.hpp` tiny and split ownership by module |
| Legacy code lingers | Delete/rename old files after cutover and remove all old call sites |
| bgfx frame pump remains throttled | Make popup-active mode advance one bgfx frame every submission-thread tick |
| Popup destroy races pending readbacks | Drain pending readbacks first, then destroy popup-scoped GPU resources |
| Mailbox drift or torn reads | Use mutex-protected full-frame copy plus generation counter, and premultiply outside the lock |
| AppKit/Cocoa calls leak onto the submission thread | Keep `NSWindow` / `NSView` / `CALayer` creation and mutation on the AppKit thread only |

---

## Recommended Order of Real Code Changes

When implementation starts, change files in this order:

1. add new popup 3D core files and mailbox types
2. add single-global-bgfx-thread ownership hooks (`reset epoch`, popup view ID reservation, popup-active frame-pump rules)
3. add new shaders and asset-copy/build rules
4. implement offscreen render path and delayed readback ring
5. implement `grim_popup_object` definition
6. simplify `popup_window.cpp` into presenter-only code with premultiplication
7. wire `pop_ui.cpp` snapshot publishing to the new renderer handoff
8. remove the sprite path entirely

This order keeps the cutover controlled and avoids mixing deletion with unproven render code too early.

---

## Final End State

After this plan is implemented, the popup system should be:

- a **single 3D object popup renderer**
- rendered offscreen through **bgfx**
- presented through a **transparent platform popup surface**
- organized into **small focused files**
- free of **sprite compatibility code**
- free of **hardcoded asset paths**
- simple enough to maintain without turning into a mini-engine

That is the intended target architecture.
