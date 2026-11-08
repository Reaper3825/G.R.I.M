# Cursor, Capture, and Input Routing

Source-reviewed November 8, 2026. This guide replaces the old visibility-gated
mouse quick reference. It describes source behavior, not a verified live
desktop/game interaction or a guarantee that background applications ignore
clicks.

## Cursor and capture

Four mechanisms must not be conflated:

| Boundary | Owner | Meaning |
|---|---|---|
| Hardware observation | `InputState` / `PlatformInput` | Reads pointer/buttons/keys; does not consume or block OS events |
| UI routing | `UIRoot` / `InputRouter` / widgets | Chooses which GRIM panel/widget receives observed input |
| Native hit-test and capture | Overlay/viewport window procedures / `PlatformWindow` | Determines native window interaction and drag capture |
| Cursor appearance | `GRIM::Input::Cursor` | Owns replacement/restoration of the OS normal cursor, not event filtering |

### Capture is always enabled

[InputState::shouldProcessMouseInput](../../core/input_parser.cpp#L27)
**always returns true**. Both `capture()` and `captureFromHWND()` set
`mouseInputEnabled = true`; they do not apply
`isPopupVisible() || shouldReceiveInputAt(...)` as a visibility gate.
Hidden panels/popups do not force mouse buttons up or suppress desktop clicks.

`capture()` records desktop pointer/buttons/keys. The UI uses
[captureFromHWND](../../core/input_parser.cpp#L79), which records window-relative
coordinates (desktop coordinates without a window), derives press/release edges
from previous button/key state, and resets per-frame wheel/text/clipboard data.
Mouse position/delta and keyboard observation continue independently of panel
visibility. [Windows PlatformInput](../../core/platform_input.cpp#L133) uses
`GetAsyncKeyState` and `GetCursorPos`/`ScreenToClient`, not visibility-based
suppression. This global-capable polling observes input while another app owns
focus; it is not an OS input-blocking hook.

## UI routing and focus

[system_tick](../../core/system_tick.cpp#L49) captures raw input, updates shared
Mouse/Key state and wake/global bindings, then sets `interactionCancelled` when
the overlay is not the active input window. UI gestures are canceled while
global bindings retain raw observation.

[UIRoot::update](../../ui/ui_root.cpp#L71) updates visible panels but suppresses
pointer/keyboard input for nonowners. Pointer selection prefers an existing
capture/popup owner; otherwise it picks the topmost visible panel under the
pointer that is not pass-through there. Keyboard ownership follows popup,
focus, then pointer panel. Clicking away can relinquish keyboard focus.
Escape with capture or explicit interaction cancellation cancels a gesture.

[InputRouter](../../ui/input/ui_input_router.hpp#L16) is UI-thread-owned, with
no OS polling. It tracks widget/panel capture, popup, focus, and wheel ownership;
frame capture remains latched after release so a later sibling cannot receive
the same click/release. Hidden/removed or no-longer-seen owners are forgotten/
canceled. Bounds, clipping, and activation remain widget responsibilities:
ordinary buttons activate on a valid release inside their hitbox, not every
hardware press. See [widgets and layout](../ui/WIDGETS_INPUT_LAYOUT.md).

Logical widget capture is mirrored to native pointer capture after panel update.
It is not synonymous with replacing the cursor or globally disabling buttons.

## Native window hit-testing and capture

The Windows overlay's
[WM_NCHITTEST](../../core/window_manager.cpp#L243) converts screen coordinates to
client space and returns:

- `HTCLIENT` if a cached panel accepts the position, **any panel is dragging/
  resizing or router capture is active**, or **the popup is visible**.
- `HTTRANSPARENT` otherwise.

[Cached panel hit-testing](../../ui/ui_root.cpp#L291) refreshes each frame,
checks pass-through rectangles before panel rectangles, and includes drag/
capture state. Popup visibility is an additional broad hit-test condition:
it is **not** restricted to the popup's own bounds in this overlay branch.
Consequently, even positions outside a panel can be claimed while popup/drag
state is active.

`HTTRANSPARENT` is native hit-test behavior, not a system-wide suppression
guarantee. Underlying desktop/apps can receive clicks according to Windows
window, thread, activation, and capture rules. Claiming a window hit does not
mean a GRIM widget must activate there. Never describe "clicks pass through" as
"no accidental desktop clicks": those are different outcomes.

[setInputPointerCapture](../../core/platform_window_win32.cpp#L341) uses
`SetCapture`/`ReleaseCapture` for the window; this retains drag delivery outside
bounds. [ViewportWndProc](../../core/platform_window_win32.cpp#L82) also captures
on button-down, releases after the final button-up, and releases on focus loss.
The overlay's `WM_MOUSEACTIVATE` returns `MA_ACTIVATE`.

[UIRoot::updateOverlayInteraction](../../ui/ui_root.cpp#L162) calls the platform
click-through helper, but the
[Windows helper](../../core/platform_window_win32.cpp#L408) is a **no-op**.
On Windows, do not cite that call as the mechanism changing overlay styles;
the actual overlay `WM_NCHITTEST` branch matters. Other platforms have their
own window implementations; these Win32 details are not cross-platform promises.
Separate native popup behavior belongs to [popup design](../ui/POPUP_3D.md).

## System cursor ownership

[Cursor](../../core/input/Cursor.hpp#L24) is a noncopyable/nonmovable reusable
appearance owner. Derived classes supply premultiplied ARGB bitmaps; dimensions
must be 1–256 pixels, pixel count must match, and the hotspot must be inside.

[Apply](../../core/input/Cursor.cpp#L150) validates/converts the bitmap, then
replaces Windows `OCR_NORMAL` with `SetSystemCursor`. A mutex-protected,
process-lifetime coordinator tracks one owner: a replacement marks the previous
owner unapplied. The OS consumes the replacement native handle.
[Restore](../../core/input/Cursor.cpp#L186) restores the user's configured cursor
scheme through `SPI_SETCURSORS` only for the current owner; an older owner cannot
undo a newer replacement. Unsupported platforms return failure from Apply.

Callers must check `Apply()`/`Restore()` and `LastError()`. Restoration can fail
and leave ownership active; the destructor attempts restoration but swallows
exceptions and cannot report a successful restore guarantee. This changes
system-wide normal-pointer appearance, not pointer position, click acceptance,
capture ownership, or desktop input authorization.

## Troubleshooting and review

- Missing widget response: check activation/cancellation, visible bounds,
  clipping, topmost panel selection, and capture/popup owner before blaming
  `mouseInputEnabled`.
- Clicks reaching desktop: inspect native hit-testing and capture. With no
  accepting UI/popup/drag, underlying app interaction can be expected—not a
  failed "mouse disabled" feature.
- Broad interception outside panels: check popup visibility and drag/capture
  state in the overlay hit-test.
- Stuck or cross-widget gesture: inspect actual press/release edges, frame
  ownership, focus loss, hidden-panel cancellation, and native release.
- Cursor scheme not restored: inspect current cursor owner and explicit
  `Restore()` result; appearance is separate from interaction state.

Safe manual QA should use a harmless desktop target, overlapping panels,
drag-out/release, focus switching, popup-visible/hidden cases, and intended
pass-through areas. Do not expect hidden GRIM UI to prevent desktop actions.
There is no current configurable `mouse_filter_mode`, `require_panel_focus`, or
`click_through_margin` contract from the historical proposal, nor a supported
performance/click-success benchmark.

Existing [UI foundation checks](../../tests/run_ui_input_foundation.ps1) cover
focused host input/widget contracts; native desktop behavior requires separate
authorized QA. No tests, binaries, captures, or input were executed here.
[Input tools](TOOLS.md#grounded-dispatch) describe the separate injection path.
