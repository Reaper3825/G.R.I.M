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
| 3 | **Transparent background remains mandatory.** The visible popup window stays a layered window with per-pixel alpha. |
| 4 | **bgfx renders offscreen only.** The final visible popup is still presented via `UpdateLayeredWindow`, not direct bgfx presentation to the popup HWND. |
| 5 | **Project-native logic only.** Use existing project dependencies (`bgfx`, `bx`, `bimg`, `stb`) and local code only. No Assimp, no external scene system, no generic model framework. |
| 6 | **Simple material model.** One albedo texture + one packed material texture. No full PBR, no shadow maps, no skeletons, no scene graph. |
| 7 | **Modular file layout.** Rendering core, material handling, mesh handling, object definition, and window presentation live in separate files to avoid bloated includes and mixed responsibilities. |
| 8 | **Plain-data architecture over inheritance.** Prefer small structs and explicit builder functions over deep abstract class hierarchies. |
| 9 | **Cross-machine paths only.** No hardcoded `D:/...` resource paths remain in the popup pipeline. |
| 10 | **Windows-first popup cutover.** The current popup system is `_WIN32`-scoped, so the first implementation targets Windows popup rendering cleanly. |

---

## Problem Statement

The current popup path is not a real 3D renderer:

- `popup_ui/popup_window.cpp` creates a transparent layered window and performs CPU-side RGBA composition.
- `popup_ui/popup_renderer.cpp` loads images on the CPU only.
- `resources/shaders/vs_sprite.sc` and `resources/shaders/fs_sprite.sc` are only sufficient for a textured 2D quad with alpha.
- Lighting is currently simulated by CPU-side mask blending rather than actual normals, camera transforms, or light equations.

That architecture cannot cleanly support a true textured 3D object with native lighting while preserving a transparent desktop background.

The correct architecture is:

1. Render the 3D object offscreen with bgfx into an RGBA target cleared to transparent.
2. Read back the rendered pixels.
3. Premultiply alpha and present through the existing layered-window mechanism.

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
pop_ui.cpp
	-> updates PopupAnimState
	-> asks Popup3DRenderer for current frame
	-> passes frame pixels to PopupWindowPresenter
	-> PopupWindowPresenter calls UpdateLayeredWindow

Popup3DRenderer
	-> builds mesh/material/object state
	-> renders offscreen via bgfx
	-> reads back RGBA pixels with alpha preserved
```

### Responsibility split

| Area | Responsibility |
|------|----------------|
| `pop_ui.cpp` | Popup loop, animation state updates, show/hide timing, render orchestration |
| popup window presenter | Window creation and pixel presentation only |
| 3D renderer | bgfx resources, view/projection, offscreen render pass, readback |
| mesh module | Vertex/index buffer data and GPU upload |
| material module | Texture load, uniform creation, shader binding |
| object definition file | The actual popup object definition (geometry/material defaults/transform defaults) |

---

## Target File Layout

The new layout should stay intentionally small.

```text
popup_ui/
	popup_window.hpp/.cpp                # layered window creation + present RGBA buffer
	popup_3d_types.hpp                  # tiny POD structs/enums only
	popup_3d_mesh.hpp/.cpp              # mesh CPU data + bgfx buffer ownership
	popup_3d_material.hpp/.cpp          # textures, uniforms, shader binding
	popup_3d_shaders.hpp/.cpp           # program load/create + shader uniform handles
	popup_3d_renderer.hpp/.cpp          # offscreen bgfx render + readback
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
- `PopupFrameBuffer`
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
- layered window presentation
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
- write final RGBA with preserved alpha

### Lighting model

First implementation uses:

- ambient term
- one directional light
- simple specular approximation

Do **not** implement full PBR or shadow maps in the first cut.

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
- readback into CPU-visible RGBA buffer

### Required render target properties

- color format with alpha
- depth buffer attached
- clear color must be transparent: alpha = 0

### Required bgfx state

- RGB write
- alpha write
- depth write
- depth test
- cull back faces
- MSAA if practical
- alpha blending enabled only if the object material actually requires it

### Important rule

The renderer must never paint a full opaque background. If it clears to opaque, desktop transparency is gone.

---

## Window Presentation Plan

`popup_window.cpp` needs to get much simpler.

### After cutover, `popup_window.cpp` should only do:

- create the popup HWND
- own the layered-window presentation code
- accept a prepared RGBA buffer
- convert RGBA to premultiplied BGRA
- call `UpdateLayeredWindow`

### It should no longer do:

- texture loading
- CPU fake lighting
- roughness/emissive mask math
- occlusion-derived shadow generation
- sprite-specific asset loading

This file becomes a presenter, not a renderer.

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

It should only prepare a simple render input struct and call the renderer.

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
	load sprite textures -> derive CPU masks -> fake light on CPU -> UpdateLayeredWindow

new:
	render 3D object offscreen with bgfx -> read back RGBA -> UpdateLayeredWindow
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
- `popup_3d_assets.hpp/.cpp`
- `popup_ui/objects/grim_popup_object.hpp/.cpp`

Deliverable:

- project builds with new modules present
- no behavior change yet

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
- readback path

Deliverable:

- renderer can produce an RGBA frame with transparent background

### Phase 4 — Define the first popup object

In `grim_popup_object.cpp` implement:

- mesh creation or mesh loading
- material asset references
- transform defaults
- simple animation tuning defaults

Deliverable:

- one stable popup object renders correctly offscreen

### Phase 5 — Replace the popup presentation input

Refactor `popup_window.cpp` so it only presents provided RGBA buffers.

Deliverable:

- layered window presentation works with renderer output

### Phase 6 — Integrate into `pop_ui.cpp`

Replace the sprite/CPU shading path with:

- animation update
- renderer call
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

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Offscreen alpha comes back opaque | Explicitly clear render target alpha to 0 and validate readback before presentation |
| Readback stalls performance | Keep popup render target small and render one object only |
| Shader asset loading fails | Add deterministic build/copy rules and startup validation with hard errors |
| Include sprawl returns | Keep `popup_3d_types.hpp` tiny and split ownership by module |
| Legacy code lingers | Delete/rename old files after cutover and remove all old call sites |

---

## Recommended Order of Real Code Changes

When implementation starts, change files in this order:

1. add new popup 3D core files
2. add new shaders and asset-copy/build rules
3. implement offscreen render path
4. implement `grim_popup_object` definition
5. simplify `popup_window.cpp` into presenter-only code
6. wire `pop_ui.cpp` to the new renderer
7. remove the sprite path entirely

This order keeps the cutover controlled and avoids mixing deletion with unproven render code too early.

---

## Final End State

After this plan is implemented, the popup system should be:

- a **single 3D object popup renderer**
- rendered offscreen through **bgfx**
- presented through a **transparent layered window**
- organized into **small focused files**
- free of **sprite compatibility code**
- free of **hardcoded asset paths**
- simple enough to maintain without turning into a mini-engine

That is the intended target architecture.
