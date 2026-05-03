# Overlay Renderer Ownership Taxonomy

`OverlayRenderer` intentionally keeps one public class/API in `ui/overlay_renderer.hpp`, but its implementation is split by ownership domain. This prevents the core renderer file from becoming the owner of unrelated systems such as icon loading, blur kernels, backdrop capture, and primitive rasterization.

## Module ownership

| File | Owns | Must not own |
|---|---|---|
| `ui/overlay_renderer.cpp` | Platform init/shutdown, frame begin/end, clip stack, dirty rect tracking | Text/font atlas, blur kernels, glass capture/cache, backdrop synthesis, primitive drawing |
| `ui/overlay_renderer_text.cpp` | UTF-8 decode, text font loading, icon font loading, STB atlas rebuild, text draw/measure | Platform frame lifecycle or glass/backdrop effects |
| `ui/overlay_renderer_primitives.cpp` | Basic software rasterization: rects, rounded rects, borders, glow, lines | Font loading, blur, desktop capture |
| `ui/overlay_renderer_blur.cpp` | Separable Gaussian blur and its scratch buffers/kernel cache | Panel composition or capture policy |
| `ui/overlay_renderer_backdrop.cpp` | Synthetic fallback backdrop/noise buffer and backdrop copy helpers | Desktop capture or native platform blur masks |
| `ui/overlay_renderer_glass.cpp` | Glass panel composition, desktop capture, distortion, cache invalidation, grain | STB/font loading or renderer lifecycle |

## Coherence rules

- `STB_TRUETYPE_IMPLEMENTATION` belongs in exactly one translation unit: `ui/overlay_renderer_text.cpp`.
- `overlay_renderer.cpp` is the only module that should own platform frame lifetime (`init`, `shutdown`, `beginFrame`, `endFrame`).
- Native platform blur ownership is decided during renderer init:
  - macOS: native blur owns the backdrop.
  - Windows/generic software path: glass capture owns per-panel backdrop.
- `overlay_renderer_glass.cpp` may call primitive and blur methods to compose the final panel, but those helpers keep their own implementation ownership.
- Cache invalidation for glass is based on panel geometry, frame cadence, backdrop sample delta, and drag/resize deferral.
