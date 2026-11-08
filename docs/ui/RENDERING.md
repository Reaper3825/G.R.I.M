# Rendering and Theme

Current source is authoritative. This guide replaces competing glassmorphism,
renderer-taxonomy, and polish recipes; it does not prescribe a second palette
or a replacement renderer.

## Module ownership

[OverlayRenderer](../../ui/overlay_renderer.hpp) has one public API with
implementation units split by responsibility:

| Implementation | Owns |
|---|---|
| [overlay_renderer.cpp](../../ui/overlay_renderer.cpp) | Platform init/shutdown, frame begin/end, clipping, dirty regions |
| [overlay_renderer_text.cpp](../../ui/overlay_renderer_text.cpp) | UTF-8/font/icon atlas loading, text drawing and measurement; sole STB font implementation |
| [overlay_renderer_primitives.cpp](../../ui/overlay_renderer_primitives.cpp) | Software rectangles, rounded shapes/borders, glow, and lines |
| [overlay_renderer_blur.cpp](../../ui/overlay_renderer_blur.cpp) | Separable Gaussian blur, scratch, and kernel cache |
| [overlay_renderer_backdrop.cpp](../../ui/overlay_renderer_backdrop.cpp) | Synthetic backdrop/noise and region copies |
| [overlay_renderer_glass.cpp](../../ui/overlay_renderer_glass.cpp) | Panel composition, desktop capture, distortion, grain, and panel caches |

Glass composition may call primitives/blur; it does not own font loading or
platform frame lifetime. Panel/window interaction is separate from rasterization.

## Rendering and theme

[UITheme](../../ui/ui_theme.hpp) owns the actual shared colors, typography,
spacing, sizes, and blur setting. The current palette is neutral charcoal with
muted blue-gray accents, not the old amber/cyan replacement tables.

Useful retained design principles are readable dark/recessed surfaces,
restrained borders/highlights, clear visual hierarchy, and consistent spacing.
Use existing tokens instead of copying old hex values into each panel.
[UIDrawHelpers](../../ui/ui_draw_helpers.hpp) supplies section/divider/widget
helpers; that does not imply every panel already uses identical layouts.

[UIPanel::drawOverlay](../../ui/primitives/ui_panel.cpp) calls `drawGlassPanel`
with theme values, a panel cache identity, and drag/resize deferral. It renders
rounded chrome dots, a centered title, and a resize grip. Maximized panels
use zero corner radius. Rounded primitives are supported: the old
"cannot draw rounded corners" limitation is obsolete.

## Glass, backdrop, and blur

`drawGlassPanel()` reads the platform blur style and distinguishes native
platform blur from software desktop capture. The software path requests
`PlatformWindow::captureDesktopBehindOverlay()` when there is an overlay handle,
blur is enabled, intensity is positive, and native blur does not own the backdrop.
The old overlay-only/no-desktop-capture restriction is not current.

The glass unit owns geometry/masks, distortion sampling, cache invalidation,
grain, and readable tint/borders. Cache refresh considers frame cadence and
backdrop sample changes; dragging/resizing can defer refresh. This is not a
new instruction to capture desktop content from unrelated widget code.

`blurRegion()` currently constructs normalized Gaussian weights and performs
horizontal/vertical passes over a clipped buffer region. Its public header's
older "box blur" comment does not change the actual implementation. Glass
composition computes its effective distortion/frost radius from the authored
theme radius and runtime blur intensity; the old constant radius-6 recipe is
not the complete panel policy.

Capture buffers, blur scratch, kernel caches, and per-panel caches have different
owners. The code includes explicit reuse behavior when capture does not yield
a fresh image. That is a description of current code, not a recommendation to
introduce new silent recovery paths.

## Authoring rules

- Keep drawing in drawing methods and widget interaction state in widgets.
- Use theme tokens/helpers; do not duplicate hover-state arrays in Settings.
- Keep input clipping/hit testing aligned with displayed geometry.
- Preserve renderer clip-stack and frame ownership when adding drawing helpers.
- Do not couple `popup_ui`'s native 3D lifecycle to overlay styling changes.
- Generic panel visibility/discovery comes through
  [panel operations](PANEL_OPERATIONS.md), not renderer draw commands.
- Use [widget/input/layout contracts](WIDGETS_INPUT_LAYOUT.md) for callback
  lifetime, pointer capture, scrolling, and coordinate transforms.

## Future design work

The old polish proposal listed tooltips, keyboard navigation, smooth transitions,
accessibility contrast targets, and advanced theming. Those are design topics,
not verified universal features or fixed 150-ms/60-FPS contracts. Reconsider
them here against actual widget behavior before adopting a requirement.

The old half-resolution/dirty-cache blur cost estimates were proposals, not
measurements of today's desktop-capture/Gaussian pipeline. Any performance or
visual-quality claim needs a named configuration, measurement, and authorized
native UI test. No runtime test or build was performed for this consolidation.
