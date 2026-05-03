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
