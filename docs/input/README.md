# Input Documentation

Input has three separate boundaries: registered tools that send desktop input,
host capture/UI routing of observed input, and ownership of the system cursor.
Use input tools only with authorization; they affect the current desktop and
focused application.

| Guide | Scope |
|---|---|
| [Tools and grounded dispatch](TOOLS.md#grounded-dispatch) | Five plugin tools, raw arguments, exact targeting, revalidation, and results |
| [Cursor and capture](CURSOR_FILTERING.md#cursor-and-capture) | Always-on capture, UI ownership, OS hit-testing/capture, and cursor lifetime |
| [UI widgets/input/layout](../ui/WIDGETS_INPUT_LAYOUT.md) | Widget hitboxes, activation, clipping, and canvas geometry |
| [UI panel operations](../ui/PANEL_OPERATIONS.md) | UI-thread requests and panel lifecycle, not synchronous input calls |

## Source boundary

Source-reviewed November 8, 2026; this consolidation ran no binaries, input
actions, captures, builds, or focused runtime tests.

[input_plugin](../../plugins/input_plugin/input_plugin.cpp#L47) owns registration
and result copying. [Host input service](../../perception/input_command_service.cpp#L5)
owns handler dispatch; perception/capture/InputController remain host-owned.
API v3.9 introduced this seam; the plugin checks the current compiled ABI, which
can be newer. See [API](../PLUGIN_API.md#input-tool-migration-v39) and
[plugin deployment](../../plugins/README.md#input-plugin).

## Current versus historical claims

These guides consolidate the input-control, intelligent-input, exact-click, and
mouse-filtering notes. Old self-learning/ActionExecutor tutorials do not define
the registered tool surface. Retained parsers or low-level scroll/double-click
methods do not create new tools.

`InputState::shouldProcessMouseInput()` now always returns true. Hidden UI does
not disable desktop mouse buttons. Widget routing and native window hit-testing
decide who responds; they do not guarantee "desktop clicks ignored." Sending
input does not verify the application's response, and raw input paths can mask
invalid keys or OS failures. Custom cursor appearance is separate from pointer
capture and event suppression.

See [physical perception](../PHYSICAL_PERCEPTION_SUBSYSTEM.md) for camera/gesture
control, rather than treating desktop injection as the physical-input contract.
