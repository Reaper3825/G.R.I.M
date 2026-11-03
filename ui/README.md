# UI Folder Layout

The UI folder is split by ownership domain so root-level files stay focused on feature panels, renderer services, and UI coordination.

## `ui/primitives/`

Reusable widget primitives live here:

- `widget.hpp` — UI-facing shim for the shared `helpers/Widget` base.
- `ui_panel.*` — base draggable/resizable panel primitive.
- `ui_button.*`, `ui_inputbox.*`, `ui_dropdown.*`, `ui_slider.*`, `ui_toggle.*` — interactive controls.
- `ui_label.*`, `ui_progress_bar.*`, `ui_scrollbox.*`, `ui_textarea.*`, `ui_graph.*`, `ui_consoleview.*` — display and content widgets.
- `ui_layout_box.*`, `ui_action_menu.*` — reusable layout/menu primitives.

Feature panels such as `ui_training_panel.*`, `ui_data_hub.*`, `ui_storage_panel.*`, and `ui_physical_environment_panel.*` include primitives via `primitives/<file>.hpp` and remain at `ui/` root for now.

## Include rule

Root-level UI panels and services should include primitives explicitly:

`#include "primitives/ui_button.hpp"`

Primitive implementation files may include sibling primitives directly and should use `../` for root UI services such as `ui_theme.hpp`, `ui_renderer.hpp`, `overlay_renderer.hpp`, `ui_focus_manager.hpp`, and `ui_root.hpp`.

## UIGrid

`primitives/ui_grid.hpp` provides a reusable `Widget` container for equal-sized
row-major tiles. Add the grid to a panel once; add its children with `addWidget`
(do not also add those children to the panel). Calendar uses seven columns with
rows fitted to its available day area.

- `setColumns(n)` fixes the column count; `setAutoColumns(minimumTileWidth)`
  derives it from the available width.
- `setSpacing(x, y)` and `setPadding(p)` control gaps and insets.
- `setRowHeight(h)` uses fixed-height rows; `setFitRows(true)` fills the viewport.
- The caller supplies position/size. `layout()` runs automatically before update
  and rendering; call it explicitly when inspecting geometry immediately.
- Hidden children do not occupy cells. Duplicate/null/self insertion is ignored.
  Remove/clear/hide cancels descendant input. Children inherit panel ownership.
- Overlay drawing and input are clipped to the grid viewport. The legacy draw
  path delegates to children without adding renderer clipping.
- Compose scrolling with UIScrollBox. Give the grid a content-sized height inside
  that scrollbox and use `getContentHeight()` after layout for fixed-height rows.
  Child coordinates are recalculated from the grid's current position each pass.

The container owns child layout, input and rendering, not tile selection or domain
state. Calendar continues to own dates, selection, labels and event counts.
