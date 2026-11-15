# Console and Settings Integration

Source-verified November 15, 2026. The current console is an overlay `ConsolePanel`,
not the former independent GDI console window and thread.

## Current source

- [ConsolePanel constructor](../../plugins/console_plugin/console_panel.cpp#L29) builds the input box
  and toolbar. Its Settings button shows the provider-owned `Settings` panel.
- [ConsolePanel::submitPrompt / startNextRequest](../../plugins/console_plugin/console_panel.cpp#L431)
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

Escape in [ConsolePanel::update](../../plugins/console_plugin/console_panel.cpp#L228) hides the console.
Panel chrome close also hides it; neither operation exits the host.

## Agent panel operations

[ui_plugin](../../plugins/ui_plugin/ui_plugin.cpp#L85) supplies `ui.list`,
`ui.inspect`, `ui.show`, `ui.hide`, `ui.bring_to_front`, and `ui.poll` through
the host panel service. For example, tool arguments for showing Settings are
`{"panel_id":"settings"}`. A pending response must be polled with its returned
`request_id`; it is not completed UI work.

The `settings` AI/admin wrapper is retired. The actual Settings panel and host
configuration services remain; see [retained-handler distinctions](../RETIRED_AGENT_TOOLS.md).

## Provider ownership

`console_plugin` owns the implementation and the `ConsolePanel` instance. It
constructs the panel during provider initialization; the `UIPanel` constructor
stages registration under that generation. The stable key remains `console`,
and the panel starts hidden, matching the former host factory default.

The host attaches, draws, routes input, publishes the `ui.panel` resource, and
detaches before plugin shutdown. The plugin destroys the panel before DLL unload.
Queued prompts are discarded at retirement; an active command is joined and its
result delivered to host session history before destruction completes. Shutdown
can therefore wait for an active command; this does not add command cancellation.

Conversation handling, session state, and display histories remain host services.
Notifications use `ConsoleNotificationTarget`, with a fresh UI-thread lookup,
without importing the plugin's concrete class. Generic `ui.*` operations remain
in `ui_plugin`. Console no longer has a bootstrap factory or a `register_ui_panel`
request. Its World Lab resource belongs to the Console Plugin room.

Native UI plugins require matching host/compiler/CRT/UI ABI and host import
libraries. Focused DLL lifecycle tests run through
`tests/run_ui_input_foundation.ps1`, with command and render services stubbed.
