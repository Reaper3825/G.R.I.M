# Console and Settings Integration

Source-verified November 8, 2026. The current console is an overlay `ConsolePanel`,
not the former independent GDI console window and thread.

## Current source

- [ConsolePanel constructor](../../ui/console_panel.cpp#L29) builds the input box
  and toolbar. Its Settings button shows the host-owned `Settings` panel.
- [ConsolePanel::submitPrompt / startNextRequest](../../ui/console_panel.cpp#L431)
  queue session-specific prompts and run `handleCommand` asynchronously.
  [handleCommand](../../commands/commands_core.cpp#L39) passes user input to
  `ai_process`; typing a tool-looking string is not automatic direct dispatch.
  The old host multi-command flow is legacy, not this entry boundary.
- [Popup window procedure](../../popup_ui/popup_window.cpp#L89) handles
  `WM_LBUTTONDOWN` by showing the `Console` panel. It does not call the former
  `GRIMConsole::showConsole()` API.
- [processPanelShortcuts](../../ui/ui_shortcuts.cpp#L29) toggles panels through
  `UIRoot`. It skips shortcuts while widget focus, popup ownership, or pointer
  capture is active.

## Access and visibility

Defaults in [InputBindings::actions](../../core/input/InputBindings.cpp#L57) are
**Ctrl+Shift+Grave** for Console and **Ctrl+Shift+F10** for Settings. They are
configurable in Settings → Controls. Bare Grave is not the documented default.
Right Ctrl is the default `wake_voice` binding;
[WakeKey::update](../../wake/wake_key.cpp#L64) initiates capture on its press.

Escape in [ConsolePanel::update](../../ui/console_panel.cpp#L228) hides the console.
Panel chrome close also hides it; neither operation exits the host.

## Agent panel operations

[ui_plugin](../../plugins/ui_plugin/ui_plugin.cpp#L85) supplies `ui.list`,
`ui.inspect`, `ui.show`, `ui.hide`, `ui.bring_to_front`, and `ui.poll` through
the host panel service. For example, tool arguments for showing Settings are
`{"panel_id":"settings"}`. A pending response must be polled with its returned
`request_id`; it is not completed UI work.

The `settings` AI/admin wrapper is retired. The actual Settings panel and host
configuration services remain; see [retained-handler distinctions](../RETIRED_AGENT_TOOLS.md).
