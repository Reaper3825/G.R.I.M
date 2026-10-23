# Canvas primitive

`ui/primitives/ui_canvas.hpp` provides `UICanvas`, a reusable overlay viewport.
It has no TextWorld, inference, document, or node-model dependencies. The existing
recursive UI source discovery in `cmake/Sources.cmake` includes it automatically.

## Navigation and coordinates

- Middle-button drag pans, retaining pointer capture outside the viewport.
- Vertical wheel zooms around the pointer, including fractional trackpad steps.
- Shift-wheel and horizontal wheel pan horizontally.
- Default zoom range is 0.1–4.0. `setZoomLimits` accepts 0.001–1000.
- `resetView` restores origin zero and scale one (subject to zoom limits).
- `fitToBounds` centers world bounds with padding measured in screen units.

The view origin is the world coordinate at the viewport's top-left. World units
equal overlay pixels at zoom one. Widget position and size are absolute overlay
coordinates during update and draw, following the existing primitive convention.
`worldToScreen`, `screenToWorld`, `visibleWorldRect`, and `isWorldRectVisible`
provide shared drawing, hit-testing, and culling geometry. Camera state belongs
to the viewport, so document coordinates never change when navigating.

## Extension example

```cpp
#include "ui/primitives/ui_canvas.hpp"
#include "core/input_parser.hpp"

class GameCanvas : public UICanvas {
    void drawCanvasContent(OverlayRenderer& renderer) override {
        drawWorldRect(renderer, {{20,20},{160,80}}, 0xFF303848);
        drawWorldText(renderer, {32,32}, "Room", 0xFFFFFFFF);
    }
    bool updateCanvasContent(const InputState& source, float dt) override {
        const auto input = routeInput(source);
        const Vec2 worldPointer = screenToWorld(input.mousePos);
        // Hit-test document geometry using worldPointer. Claim pointer capture
        // for node drags, release it on completion, and return true while handled.
        return false;
    }
};
```

Set the canvas rectangle in the owning panel's layout, call `update`, then
`drawOverlay`. Drawing runs under an intersected renderer clip, including the
background and adaptive grid; the clip is restored if a content hook throws.
World drawing helpers scale geometry, text, and line widths. Raw renderer calls
remain screen-space calls; there is no implicit renderer transform.

`updateCanvasContent` receives viewport-clipped **screen-space** input before
navigation. Leaf content calls `routeInput`; a container forwards this input to
children, which each route it themselves. Use `screenToWorld` for model hit tests.
Do not forward an input state already suppressed by the canvas to child widgets.
Child pointer capture or wheel consumption prevents navigation from stealing it.
Returning true also reserves that frame's navigation. An established middle-button
pan skips content updates until release. Escape/focus-loss cancellation comes
from the root router; `cancelInput` clears the canvas pan immediately.

The primitive does not automatically transform ordinary child widgets. A node
editor should render its world geometry using the helpers and explicitly position
any text editors or menus in screen coordinates. Node selection, connections,
undo/redo, serialization, and training-panel placement are separate layers.

## Validation

Run `powershell -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1`.
This uses the existing MSVC toolchain for host-only UI tests, without launching
GRIM or any training/inference runtime. Canvas coverage includes transform
round-trips, anchored zoom and limits, fit/culling, fractional wheel input,
ancestor clipping, captured pan/release outside the viewport, cancellation,
hidden state, content gesture priority, drawing hooks, and nested wheel ownership.
Rendering uses test sinks; native visual behavior still requires application QA.
