# Widgets, Input, and Layout

This is the shared contract for buttons, panel chrome, containers, Settings
refresh, and the reusable canvas. It replaces overlapping fix summaries and
the obsolete proposed generic-container migration.

## Buttons and panel chrome

[UIButton::update](../../ui/primitives/ui_button.cpp) consumes routed
`InputState`. It acquires pointer capture on a press within its five-pixel
expanded hitbox; hover uses exact visual bounds. Activation requires an actual
mouse-release edge within the expanded hitbox. Lost mouse-down state without
that release cancels activation.

Capture is released before the callback. The inside/release condition is tested
before callback execution, so the old "always call the callback regardless of
inside state" Settings fix must not be restored.
[InputRouter](../../ui/input/ui_input_router.hpp) latches frame ownership so
releasing capture does not deliver the same event to a later sibling.
[InputState::captureFromHWND](../../core/input_parser.cpp) derives platform
press/release edges; widgets should not add unrelated hardware polling paths.

[UIPanel::handleChromeButtons](../../ui/primitives/ui_panel.cpp) closes on the
press edge, unlike a normal button's release callback. Hiding/minimizing forgets
input ownership; hiding also clears panel dragging/resizing. Visibility changes
can cancel interactions rather than promising completion after layout changes.
The Console Settings button opens the existing host Settings panel; see
[Console](CONSOLE.md) for access and [panel operations](PANEL_OPERATIONS.md)
for model-selectable operations.

## Container and Settings ownership

| Owner | Current responsibility |
|---|---|
| [UIRoot](../../ui/ui_root.hpp) | Root panels, focus/Z-order, keyed requests, and owning-thread work pump |
| [UIPanel](../../ui/primitives/ui_panel.hpp) | Panel children, chrome, drag/resize, and content rectangle |
| [UIScrollBox](../../ui/primitives/ui_scrollbox.hpp) | Child collection, content height/offset, wheel ownership, clipping, and `autoLayoutChildren()` |
| [UILayoutBox](../../ui/primitives/ui_layout_box.hpp) | Existing layout primitive |
| [UISettingsMenu](../../ui/ui_settings_menu.cpp) | Tab-specific construction/layout, current scroll content, Save/Cancel, and deferred refresh |

The current panel/scrollbox classes provide child APIs directly. There is no
implemented generic `UIContainer` base to migrate every widget into.
The old plan's index-based Coqui speaker example and "delete all widget vectors"
recipe do not describe the current tabbed Settings source.

`UISettingsMenu::update()` handles a requested `needsWidgetRefresh` before its
normal replacement-widget update path, calls `createWidgets()`, and returns for
that frame. Callbacks should request that refresh rather than rebuild a
collection while it is being traversed.

`UIScrollBox::update()` supplies clipped input to children and uses displayed
scroll geometry, restoring their local positions after update. Draw and input
must agree on clipping/offset; a parent must not update a child a second time
through an unrelated manual loop.

## Historical fix notes

Copying a vector of `shared_ptr<UIButton>` copies pointers, not button objects.
It does not by itself lose widget state. Callback-driven collection replacement
is a separate lifetime/iteration problem.

The October 2025 panel-close/input-filter log and old 10%-100% click-success
tables were historical claims, not verified current benchmarks. Archived
[Settings replacement snippets](../legacy/settings_menu_update_FIXED.cpp)
must not be pasted over the current tabbed Settings implementation.

## Canvas primitive

[UICanvas](../../ui/primitives/ui_canvas.hpp) is a reusable overlay viewport,
independent of TextWorld, inference, document storage, and node models.

### Navigation and coordinates

- Middle-button drag pans, retaining capture outside the viewport.
- Vertical wheel zooms around the pointer, including fractional trackpad steps.
- Shift-wheel and horizontal wheel pan horizontally.
- Default zoom range is 0.1-4.0; `setZoomLimits` accepts 0.001-1000.
- `resetView` restores origin zero and scale one, subject to zoom limits.
- `fitToBounds` centers world bounds with screen-unit padding.

The origin is the world coordinate at the viewport top-left; world units equal
overlay pixels at zoom one. `worldToScreen`, `screenToWorld`, `visibleWorldRect`,
and `isWorldRectVisible` share drawing, hit-test, and culling geometry.
Camera navigation does not mutate stored document coordinates.

Set the canvas rectangle in its owning layout, call update, then draw.
Content drawing runs under an intersected renderer clip, restored even if a
hook throws. World helpers scale geometry/text/line widths; direct renderer
calls remain screen-space calls with no implicit transform.

`updateCanvasContent` receives viewport-clipped screen-space input before
navigation. Leaf content routes that input; containers forward it to children
which route themselves. Use `screenToWorld` for model hit tests.
Do not forward already-suppressed input to children.

Child capture or wheel consumption prevents canvas navigation from stealing
the event. Returning true also reserves navigation for that frame. Established
middle-button pan skips content updates until release; root Escape/focus-loss
cancellation clears pan via `cancelInput`.

The primitive does not transform ordinary child widgets automatically. Render
node geometry with world helpers and position editors/menus explicitly in
screen coordinates. Selection, edges, undo/redo, serialization, and model
inspection belong to feature layers, not the canvas.

### Extension example

```cpp
#include "ui/primitives/ui_canvas.hpp"

class GameCanvas : public UICanvas {
    void drawCanvasContent(OverlayRenderer& renderer) override {
        drawWorldRect(renderer, {{20,20},{160,80}}, 0xFF303848);
        drawWorldText(renderer, {32,32}, "Room", 0xFFFFFFFF);
    }
};
```

## Troubleshooting and verification

Check routed/clipped input, panel identity, true press/release edges, capture
ownership, callback lifetime, deferred refresh, and coordinate conversion.
Distinguish intended drag-out cancellation from a missed event.

From the repository root,
[run_ui_input_foundation.ps1](../../tests/run_ui_input_foundation.ps1) supplies
host-only focused checks for input/panel/canvas contracts with drawing sinks.
Native visual rendering remains a separate QA check. This documentation-only
consolidation did not build or launch the application or run those binaries.
