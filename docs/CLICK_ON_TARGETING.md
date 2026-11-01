# Grounded click_on targeting

`click_on` is supplied by the Input Plugin through API v3.9. Its retained host
handler is in `commands/commands_click_on.cpp`; target resolution is in
`perception/click_target.cpp`, with local capture/input wiring in
`perception/click_target_host.cpp`.

The plugin preserves its full result, including failure codes and structured
coordinates, through the host input service. See the
[input plugin](../plugins/README.md#input-plugin) for build/deployment and
[API contract](PLUGIN_API.md#input-tool-migration-v39) for permissions/lifetime.

## Requests and matching

- `click_on text Submit` locates an exact OCR text region.
- `click_on Submit` uses the same default text mode.
- `click_on object save-button` locates a native UI control by its exact name,
  automation ID, or role. For example, `object button` succeeds only if exactly
  one actionable control matches that role.

Selectors ignore ASCII case; labels ignore ASCII case and normalize whitespace.
UTF-8 labels otherwise compare by their byte content. An empty selector/label is
invalid. Substrings and fuzzy labels are not used to guess coordinates.

The default digital OCR provider supplies line bounds. Text mode therefore
requires a whole recognized region label, not an arbitrary word within a longer
line. If the provider cannot locate that label precisely, the request fails.
Native control targeting is useful for labels/IDs that OCR groups into a longer
line. Object mode is now explicitly native UI-control targeting, not YOLO or
general visual-object recognition: the previous legacy object path never filled
its detection vector. Unsupported controls/platforms fail explicitly rather than
falling back to a screen-center click.

## Observation and geometry

Each request makes a fresh local Windows active-window capture. Text recognition
uses those exact pixels; native control inspection is correlated with the same
capture metadata. The request does not reuse the legacy string-only visual
context, recapture for unrelated analysis, start a model, or write memories.

OCR rectangles are frame-local. Their centers are translated by the capture's
desktop origin, including negative coordinates. Native UI rectangles are already
physical desktop coordinates and are not offset or DPI-scaled again. A matching
rectangle must be nonempty and fully inside the captured region.

The resolver rejects unsuccessful/nonlocal captures, captures older than five
seconds, unavailable OCR/control providers, truncated or mismatched native
observations, low-confidence OCR (below 0.5), and disabled/offscreen/password
controls. Multiple matching regions/controls are ambiguous; the first match is
never selected arbitrarily. Invalid or blocked duplicate matches also prevent a
unique actionable target from being established.

Immediately before input, the host verifies that the foreground window handle
and window rectangle still match the capture. This narrows the observation/input
gap; it cannot guarantee that application content remains unchanged between
observation and input delivery. Content changes inside an unchanged window still
require subsequent observation to assess.

## Input and results

`InputController::clickAtChecked` verifies cursor movement and position, then
checks that Windows accepts both button-down and button-up events. A rejected or
partial send is failure; after a partial send it attempts to release the button.
Some input may already have occurred on a partial failure. Existing direct mouse,
keyboard, typing, and legacy `clickAt` behavior is otherwise unchanged.

`ClickTargetResult` carries success, an error/message, and physical desktop
coordinates on success. `cmdInputClickOn` propagates failures and includes
`input_sent`, `application_response_verified:false`, target kind/label, and
desktop coordinates in structured successful output. Success establishes input
acceptance, not that a button activated or the user's larger task completed.

The action executor's direct text-click and targeted-submit branches also honor
this result. A failed targeted submit click does not fall through to an Enter
keypress. The unrelated legacy no-target Enter fallback is unchanged.

Common failures:

| Error | Meaning |
|---|---|
| `ERR_INVALID_ARGS` | Empty label/selector or invalid button |
| `ERR_INPUT_NOT_INITIALIZED` | Command has no host perception manager |
| `ERR_INPUT_UNSUPPORTED` | No input adapter for this platform |
| `ERR_INPUT_OBSERVATION_UNAVAILABLE` | Missing, failed, incomplete, or uncorrelated capture/provider evidence |
| `ERR_INPUT_TARGET_NOT_FOUND` | No exact bounded label match |
| `ERR_INPUT_TARGET_AMBIGUOUS` | More than one matching target |
| `ERR_INPUT_TARGET_NOT_ACTIONABLE` | Invalid bounds/confidence or a blocked control |
| `ERR_INPUT_STALE_TARGET` | Observation aged out or foreground/geometry changed |
| `ERR_INPUT_DISPATCH_FAILED` | Cursor positioning or complete button delivery failed |
| `ERR_INPUT_TARGETING_FAILED` | Unexpected observation/dispatch exception |

## Validation

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_click_target.ps1
```

The suite links the real resolver, command, and action executor against simulated
observations/input and a test host manager. It checks precise bounds, negative
origins, DPI handling, ambiguity, invalid/stale/remote observations, unavailable
providers, control restrictions, partial input, and failure propagation. The
production native adapter, perception context, and input controller receive
syntax checks. Installed MSVC/OpenCV dependencies are reused. No desktop capture,
mouse/keyboard input, model execution, application build, or installation runs.
