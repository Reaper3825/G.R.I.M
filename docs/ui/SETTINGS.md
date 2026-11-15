# Settings provider

`plugins/settings_plugin` owns `UISettingsMenu`, its widgets, callbacks, and
pending edits. The provider is **Settings Plugin**, and its panel key remains
`settings`. General, Voice, Audio, Vision, UI, Preferences, GeoSpatial, Memory,
Controls, and Intents remain tabs within this one panel.

The plugin constructs the panel during initialization. The base constructor
stages it under the provider generation; commit attaches it and publishes its
`ui.panel` resource. The host handles drawing, input, visibility, and synchronous
detachment before plugin shutdown destroys the instance. It starts hidden.

Configuration loading/saving and runtime side effects remain host services in
`settings/`. Input bindings and native widgets also use the existing host
implementation. The plugin imports these C++ interfaces, requiring a matching
host compiler, CRT and UI ABI. Callbacks passed to configuration application
are synchronous and are not retained past the call.

Save & Close persists pending changes, applies runtime effects, and hides the
panel. Cancel reloads configuration and hides it. Provider unload discards pending
edits without saving; keybind widget destruction releases active shortcut capture.
Reload constructs a fresh instance and does not preserve unsaved panel state.

The bootstrap Settings factory and UI Plugin's `ui.settings` implementation
request are removed. UI Plugin still owns generic panel operation tools. The
retired `settings` agent wrapper remains retired. World Lab's Settings resource
now belongs to the Settings Plugin room.

`tests/run_ui_input_foundation.ps1` builds a focused native DLL fixture against a
test host. It checks tab rendering, Save/Cancel service calls, capture release,
replacement, queued activation cancellation, permission denial, and configuration
load failure. Configuration and OS effects are stubbed; this is not an application
runtime test and does not modify real settings.
