# Input tools — API v11

[input_plugin](../../plugins/input_plugin/README.md) owns four Debug tools:

| Tool | Raw arguments |
|---|---|
| `move_mouse` | signed `x y` or `x,y` desktop coordinates |
| `click` | `left`, `right`, `middle`; empty means left |
| `type` | nonempty raw text; legacy 15 ms typing delay |
| `key` | key name or plus-separated combination, e.g. `ctrl+c` |

All require INPUT permission and a current provider. Calls serialize inside
the DLL. Native Windows device execution is provider-owned; no host forwarding
or perception initialization is required. Unknown keys/buttons fail, as do OS
send/move failures. Success is OS acceptance, not application verification.
Non-Windows execution is unavailable. Typing retains byte-oriented limitations.

`click_on` is retired. Pure targeting geometry and checked delivery helpers are
retained under `core/input` for focused tests; they are not registered tools.
See the provider README for lifecycle and validation commands.
