# Input Tools and Grounded Targeting

Source-reviewed November 8, 2026. Authorized desktop control only. Review focus,
target, and consequences before requesting input, especially destructive actions.

## Commands

[input_plugin registration](../../plugins/input_plugin/input_plugin.cpp#L47)
supplies five canonical tools, not duplicate built-ins in `core_plugin`.
Their argument format is **raw text**, not JSON:

| Tool | Supported arguments / example | Contract |
|---|---|---|
| `move_mouse` | `<x> <y>` or `<x>,<y>`; `-1200 300` | Two signed integer desktop coordinates, including negative monitor origins |
| `click` | `[left\|right\|middle]`; `left` | At current pointer; empty argument means left |
| `type` | `<text>`; `Hello world` | Verbatim nonempty text, including surrounding whitespace; requests 15 ms per-character delay |
| `key` | `<key>` or `<key+key>`; `ctrl+c` | Host key names; plus-separated combinations trim whitespace around components |
| `click_on` | `[text\|object] <exact label>`; `text Submit` | Unique exact OCR region or native UI control in the active window |

All tools require INPUT permission and carry confirmation metadata. `click_on`
also requires SYSTEM for perception/targeting; UI permission alone is
insufficient. Descriptors are classified `GRIM_TOOL_USAGE_DEBUG`; registration
does not imply normal model-catalog selection.
[handleCommand](../../commands/commands_core.cpp#L39) routes typed text to the
model, not directly to these handlers. Natural-language action phrases or
retained NLP parsers do not guarantee tool dispatch.

[Raw handlers](../../commands/commands_perception.cpp#L13) retain permissive
behavior: coordinate parsing does not require the entire string to be consumed,
and button/key interpretation is not comprehensively validated.
[Host key mapping](../../perception/perception_context.cpp#L1590) supports
letters/digits, F1–F12, Enter/Return, Esc/Escape, Tab, Space, Backspace,
Delete/Del, Home/End, PageUp/PgUp, PageDown/PgDn, arrows, Shift, Ctrl/Control,
Alt, and Win/Windows. Unknown single keys can merely log an error; unknown combo
components can be discarded while valid components still execute.
Direct button mapping can select left for an unknown name.

[InputController](../../core/input/InputController.cpp#L10) uses void raw
movement/click/key paths that do not check OS acceptance. Typing uses byte-wise
`VkKeyScanA`, skips unsupported characters, and is not a Unicode-text guarantee.
Non-Windows raw methods include no-op implementations, so successful raw command
acknowledgments do not establish OS delivery on every platform.
No separate scroll, double-click, form-submit, automatic-learning, app-opening,
or "undetectable/hardware input" tool is registered by this plugin.

## Grounded dispatch

```text
registered input_plugin callback
  → synchronous host input_service → ExecuteInputCommandService
  → retained host handler → fresh local targeting (click_on only)
  → InputController → complete result copied back to plugin writer
```

[API v3.9](../../core/plugin_api.hpp#L419) introduced the host service seam;
[plugin initialization](../../plugins/input_plugin/input_plugin.cpp#L47) checks
same major version, at least the **current compiled `GRIM_API_VERSION`**, and a
non-null service. "Introduced in v3.9" does not mean any v3.9 host can load a
plugin compiled against a newer table.

[api_input_service](../../core/plugin_api_impl.cpp#L1704) requires an accepting
provider and permissions, copies at most 65,536 argument bytes, and calls
[ExecuteInputCommandService](../../perception/input_command_service.cpp#L5)
without recursive registry/console dispatch. Arguments are borrowed only during
the call. The host returns `grim-input-result` version 1 JSON; the plugin copies
it and releases the host string with `string_free`. Success, messages, errors,
voice, colors, structured results, and encoded exceptions are preserved.
Null service output becomes `ERR_INPUT_SERVICE` and can mean denial, invalid
arguments, stopping/unpublished provider, or API failure—not necessarily missing
perception context. Encoded exceptions are rethrown for registry exception policy.

Calls remain **synchronous**, including typing delays and fresh observation.
There are no input job IDs, workers, pending states, or `ui.poll` protocol here.
Active-call gates keep in-flight handlers alive through unload. Loading/unloading
the plugin does not initialize/destroy host capture/input services.
Missing host perception context yields `ERR_INPUT_NOT_INITIALIZED`.

## Exact click requests and geometry

[cmdInputClickOn](../../commands/commands_click_on.cpp#L15) supports:

```text
click_on Submit
click_on text Submit
click_on object save-button
click_on object button
```

Text is the default. Prefixes ignore case; target labels ignore ASCII case and
normalize whitespace. Otherwise UTF-8 labels compare by byte content. Empty/NUL
labels are invalid. Matching is exact, not substring, fuzzy, or semantic.
`object` means a native control's name, automation ID, or role, not YOLO/general
image detection. A role such as `button` succeeds only when uniquely actionable.
The registered `click_on` payload has no button option; the host default is left.

[Default OCR](../../perception/digital/DigitalOcrProvider.cpp#L96) returns text-line
bounds: a target must match a whole recognized region, not an arbitrary word
inside a longer line. Use an exact native control ID/name when OCR groups text
too broadly. Unsupported controls/platforms do not fall back to guessed
screen-center coordinates.

[ClickOnObservedTarget](../../perception/click_target_host.cpp#L12) makes a fresh
local Windows foreground-window capture per request. OCR uses those exact
pixels; native inspection is correlated with the same metadata. This path does
not reuse string-only visual context, start a model, perform unrelated
recaptures, or write memories.

[ResolveClickTarget](../../perception/click_target.cpp#L29) requires:

- Successful native/local Windows capture, valid source bounds, and age no more
  than five seconds; future/zero timestamps fail.
- Text: available OCR, exact label, confidence **0.5–1 inclusive**, nonempty
  rectangle entirely inside the frame. The rectangle is frame-local; its
  center is translated by the desktop origin, including negative coordinates.
- Object: successful complete/nontruncated native observation, matching capture,
  no detected target change, and a uniquely matching enabled, onscreen,
  non-password control entirely inside the source region.
  Native rectangles already use physical desktop coordinates: do not apply
  the origin or DPI scale a second time.
- No invalid or blocked duplicate label; these prevent proving an actionable
  target even if another valid match exists. Multiple valid matches are
  ambiguous, never "take the first."

## Revalidation and results

Immediately before dispatch,
[sourceStillCurrent](../../perception/click_target_host.cpp#L54) checks that the
foreground window handle and its rectangle still match the capture.
[ExecuteClickTarget](../../perception/click_target.cpp#L91) rejects a changed
source before sending input. This narrows, but cannot eliminate, the
observation/input race: content can change within an unchanged window.

[clickAtChecked](../../core/input/InputController.cpp#L36) verifies cursor
movement/position and that Windows accepts both button events.
[SendCheckedClick](../../core/input/checked_click.hpp#L11) fails on rejected or
partial delivery and attempts button release after a one-event send.
**Some input may already have occurred on partial failure**; do not blindly
retry destructive actions.

Successful `cmdInputClickOn` output contains `input_sent:true`,
`application_response_verified:false`, `target`, `target_kind`, `desktop_x`,
and `desktop_y`. Success means checked input acceptance, not a button's
activation or completion of the larger task. Obtain a **fresh observation
after dispatch** to verify the intended application effect.

The [host input command service](../../perception/input_command_service.cpp)
propagates targeting errors without retrying or substituting another input
action. The uncalled ActionExecutor text parser and submit/Enter shortcut were
removed on November 8, 2026. Grounded clicks remain available through `click_on`.

| Error | Meaning / next step |
|---|---|
| `ERR_INVALID_ARGS` | Empty/invalid selector or label; correct the request |
| `ERR_INPUT_SERVICE` | Host API unavailable/denied/invalid/stopping; check provider and permissions |
| `ERR_INPUT_NOT_INITIALIZED` | Missing host perception manager |
| `ERR_INPUT_UNSUPPORTED` | Grounded input adapter unavailable for the platform |
| `ERR_INPUT_OBSERVATION_UNAVAILABLE` | Missing, failed, nonlocal, incomplete, or uncorrelated evidence |
| `ERR_INPUT_TARGET_NOT_FOUND` | No exact bounded match; inspect the whole OCR label or native ID |
| `ERR_INPUT_TARGET_AMBIGUOUS` | Multiple matches; use a more specific exact label/ID |
| `ERR_INPUT_TARGET_NOT_ACTIONABLE` | Invalid bounds/confidence or a blocked matching control |
| `ERR_INPUT_STALE_TARGET` | Capture too old or foreground/geometry changed; reobserve |
| `ERR_INPUT_DISPATCH_FAILED` | Positioning or complete button delivery failed; inspect before retry |
| `ERR_INPUT_TARGETING_FAILED` | Observation/dispatch exception |

Existing [targeting checks](../../tests/run_click_target.ps1) cover simulated
observations/input, geometry, staleness, ambiguity, partial delivery, and
propagation; [plugin checks](../../tests/input_plugin_tests.cpp) cover metadata/
service behavior. They were not run for this documentation-only task.
See [cursor/capture](CURSOR_FILTERING.md#cursor-and-capture) for observed UI input,
not injected tool input.
