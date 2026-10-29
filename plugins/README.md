# Native GRIM tool plugins

Use the standalone build to produce API v3 plugin libraries without configuring
or building GRIM:

    cmake -S plugins -B .codex-build/plugins-standalone -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/plugins-standalone --config Release --target osint_plugin network_test_plugin --parallel 2

Existing installed dependencies are required; this build installs nothing.
The output folder is .codex-build/plugins-standalone/plugins by default.
GRIM_PLUGIN_OUTPUT_DIR can override it. Each plugin subdirectory is also a
standalone CMake project.

Deploy a completed native library into the application's plugins folder.
The host reconciles this folder about once a second, including removals.
A replacement that fails initialization or registry validation leaves the
working plugin loaded. Replacements must retain the provider name.

The loader uses private copies, so Windows does not lock deployed source DLLs.
Do not copy a library while its compiler/linker is still writing it. Publish a
completed file by rename where possible. Keep non-plugin dependencies outside
the scanned top-level folder.

See ../docs/PLUGIN_API.md for API ownership, reload behavior, limitations, and
focused verification. The bundled core_plugin.cpp remains part of the host.

## Filesystem plugin

`filesystem_plugin` supplies `pwd`, `cd`, `ls`, `mkdir`, and `rm`. These commands
are no longer registered by the host. `clean` and `help` come from `utility_plugin`.
The plugin depends only on the API header and the C++ standard library.

Build only this plugin with the existing toolchain:

    cmake -S plugins/filesystem_plugin -B .codex-build/filesystem-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/filesystem-plugin --config Release --target filesystem_plugin --parallel 2

Validate the real DLL through the loader/API/registry in a temporary directory:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -FilesystemPluginDirectory .codex-build/filesystem-plugin/plugins

Deploy `.codex-build/filesystem-plugin/plugins/filesystem_plugin.dll` into the
GRIM root's `plugins/` folder. A host built from the migrated source discovers
all five tools automatically. An older host that still registers these commands
will reject the plugin due to tool ID collisions; rebuild the host separately
before testing inside the application.

The plugin starts in the process working directory each time it loads. `cd`
changes only its private working directory, shared by its five tools, and reload
resets that directory. Paths may be absolute or relative, may contain spaces,
and may have enclosing double quotes. No shell expansion is performed. `ls`
lists the private working directory; it ignores arguments, as before. `mkdir`
creates one directory, and `rm` removes one file, symlink, or empty directory.
Deletion is never recursive. `cd` rejects regular files. Filesystem errors are
returned as failed tool results. Policy flags are metadata; direct invocation
does not enforce confirmation or sandbox native filesystem access.

## Application plugin

`application_plugin` supplies `open` and `search` using API v3.1. Command parsing,
result text, error codes, voice, colors, and confirmation policy match the original
host handlers. `open` passes its argument unchanged; `search` trims surrounding
whitespace and constructs the original Google search URL without new encoding.
The host supplies current configured error messages and the native OS opener.
The plugin imports no host C++ symbols. Non-Windows stubs remain unchanged.

    cmake -S plugins/application_plugin -B .codex-build/application-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/application-plugin --config Release --target application_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -ApplicationPluginDirectory .codex-build/application-plugin/plugins

Deploy `.codex-build/application-plugin/plugins/application_plugin.dll` into
GRIM's `plugins/` folder. Rebuild the host separately for API v3.1 and removal of
its old registrations. The new host continues to load v3.0 plugin binaries;
the application plugin requires v3.1. The compatibility test uses a controlled
native launch sink and does not open applications or browsers.

## Utility plugin

`utility_plugin` supplies `clean` and `help`. Their original complete result
payloads are preserved: message, success/error code, category, voice, and color.
Both still ignore arguments. `help` retains a static command list, with retired
`ai_backend` and `reloadnlp` entries removed; use `list_tools` for current registrations.
`clean` retains the original response-only handler; this migration adds no UI
clearing operation. Neither tool requires access to host state or permissions.

    cmake -S plugins/utility_plugin -B .codex-build/utility-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/utility-plugin --config Release --target utility_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -FilesystemPluginDirectory .codex-build/filesystem-plugin/plugins -UtilityPluginDirectory .codex-build/utility-plugin/plugins

Deploy `.codex-build/utility-plugin/plugins/utility_plugin.dll` into GRIM's
`plugins/` folder. The host must be rebuilt from the migrated source to remove
its old `clean`/`help` registrations. An older host rejects the colliding batch
without losing its working commands. The compatibility test checks exact results,
policy metadata, old-host collisions, and removal through folder reconciliation.

## Tool inspection plugin

`tool_inspection_plugin` supplies `list_tools`, `tool_info`, and `tool_stats` using
API v3.2's read-only registry snapshots. The original formatting and argument
behavior are preserved, including exact category matching, alias/display-name
lookup, and the existing statistics integer-prefix parsing. The plugin registers
usage, parameter, and example metadata and requires no permissions or confirmation.

    cmake -S plugins/tool_inspection_plugin -B .codex-build/tool-inspection-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/tool-inspection-plugin --config Release --target tool_inspection_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -ToolInspectionPluginDirectory .codex-build/tool-inspection-plugin/plugins

Deploy `.codex-build/tool-inspection-plugin/plugins/tool_inspection_plugin.dll`
into GRIM's `plugins/` folder. Rebuild the host separately for API v3.2 and removal
of its old registrations. The new host still supports v3.0/v3.1 plugin binaries.
The plugin imports no host C++ symbols; snapshots contain metadata, not handlers.
The snapshot codec is shared as `MMO/Core/ToolDescriptorJson.hpp`.

Tests cover complete metadata, alias/display-name lookup, unchanged result fields,
statistics limits and errors, large entries/results, copy isolation, lifecycle
metadata, and folder removal. API v3.2 permits result messages up to 16 MiB so
catalog listings can exceed 64 KiB; other result field limits stay unchanged.
World Lab object population remains a separate integration step.

## Aliases plugin

`aliases_plugin` supplies `alias list`, `alias info`, and `alias refresh` using
API v3.3. It accesses the existing host application-alias store; it does not
create a second store, scanner, persistence layer, or background refresh thread.
These aliases are separate from ToolRegistry's command aliases.

    cmake -S plugins/aliases_plugin -B .codex-build/aliases-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/aliases-plugin --config Release --target aliases_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -AliasesPluginDirectory .codex-build/aliases-plugin/plugins

Deploy `.codex-build/aliases-plugin/plugins/aliases_plugin.dll` into GRIM's
`plugins/` folder. Rebuild the host separately for API v3.3 and removal of the
old command registrations. Earlier v3.x plugin binaries remain supported by the
new host. The plugin imports no host C++ symbols and only refresh requires
filesystem permission and confirmation metadata.

Original messages, error codes, categories, voice, colors, and argument behavior
remain unchanged. In particular, unknown alias info normally succeeds with a
"not found in aliases" message. Refresh retains the existing wrapper's fixed
success response when the host refresh function returns normally, even for busy
or failed returned statuses; thrown standard exceptions produce the original
failure result. Tests exercise these behaviors using a controlled host fixture,
without scanning actual application folders or writing the real alias database.
Unloading the DLL removes its commands and leaves host alias state intact.

## System information plugin

`system_info_plugin` supplies `system_info` using API v3.4. It formats a full
snapshot from the existing host hardware detector. As before, each invocation
detects fresh hardware; it does not substitute or overwrite the startup cache.
Output text, ignored arguments, result fields, backend reporting, and error
behavior are preserved. No host C++ symbols are imported by the plugin.

    cmake -S plugins/system_info_plugin -B .codex-build/system-info-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/system-info-plugin --config Release --target system_info_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -SystemInfoPluginDirectory .codex-build/system-info-plugin/plugins

Deploy `.codex-build/system-info-plugin/plugins/system_info_plugin.dll` into
GRIM's `plugins/` folder. Rebuild the host separately for API v3.4 and removal
of the old registration. Earlier v3.x binaries remain supported by the new host.
The tool requires system-information permission, is informational, and does not
require confirmation. The shared snapshot codec is
`MMO/Core/HardwareInventoryJson.hpp`.

Tests use controlled hardware fixtures, including GPU backend flags, monitor
geometry, audio/network/voice fields, timestamps, detection errors, fresh reads,
and startup-cache isolation. They also test exact command results and DLL removal,
without probing actual hardware or starting the application.

## Timers and memory plugins

`timers_plugin` supplies `set_timer`; `memory_plugin` supplies `remember`,
`recall`, and `forget`. Both use API v3.5 host services; neither imports host C++
symbols or owns timer/storage state. Rebuild the host separately to expose v3.5
and remove the former registrations. Older v3.x plugin binaries remain supported.

    cmake -S plugins/timers_plugin -B .codex-build/timers-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/timers-plugin --config Release --target timers_plugin --parallel 2
    cmake -S plugins/memory_plugin -B .codex-build/memory-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/memory-plugin --config Release --target memory_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -TimersPluginDirectory .codex-build/timers-plugin/plugins -MemoryPluginDirectory .codex-build/memory-plugin/plugins

Deploy each DLL from its build's `plugins/` directory into GRIM's `plugins/` folder.
Timers require SYSTEM permission; memory commands require MEMORY. Recall remains
informational; set_timer, remember, and forget retain confirmation metadata.

Timers preserve integer-prefix parsing and enqueue native host timers. Memory
preserves manual-record fields, rotation staging, search defaults, first-hit
recall, error messages, voice, colors, and the legacy forget loop. Remember does
not bypass rotation for immediate durable storage. Pending timers and memory
survive DLL unload. The new services do not use the old stub memory API slots.

**Existing limitations are unchanged:** forget checks and flushes but does not
delete records. The current source has no caller for `checkExpiredTimers()`, so
automatic timer delivery is not introduced by this migration. The tests invoke
the host expiry checker directly without waiting for a real timer.

The focused suite links the real rotation and storage engine, uses a private
temporary database, and checks persistence by reloading it. It does not touch the
user's memory store. The existing generated memory header is copied into the
test include directory; no schema regeneration or dependency installation occurs.

## Perception plugin

`perception_plugin` supplies `perception_what_see`, `perception_read_text`, and
`perception_detect_objects` through API v3.6's SYSTEM-permission observation
service. The existing host retains capture, OCR, detection, live context, model
and subsystem ownership. Mouse/keyboard commands remain in the host for their
separate migration. The plugin depends only on the API, shared result helper,
and installed nlohmann_json; it imports no host C++ or OpenCV symbols.

    cmake -S plugins/perception_plugin -B .codex-build/perception-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/perception-plugin --config Release --target perception_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -PerceptionPluginDirectory .codex-build/perception-plugin/plugins

Deploy `.codex-build/perception-plugin/plugins/perception_plugin.dll` into GRIM's
`plugins/` folder. Rebuild the host separately for API v3.6 and removal of the
old registrations. Earlier v3.x plugins remain supported by the new host.

Commands preserve their original availability/error responses, ignored arguments,
OCR defaults, and literal message/voice text. Empty text and error-looking host
text still use the old success result. They remain informational with no
confirmation requirement. Loading/unloading performs no capture or perception
initialization and leaves host context intact.

Tests use controlled host services, including 70-KB text/voice results and both
exception paths; no real capture, OCR, object detection, or model server runs.
Existing OpenCV headers are used only for a syntax check of the remaining host
input-command source. API v3.6 expands the voice result limit to 16 MiB to match
messages; it does not truncate observations.

## Voice plugin

`voice_plugin` supplies `voice`, `voice_stream`, `test_tts`, `list_voices`,
`test_sapi`, `tts_device`, `nevermind`, `create_embedding`, `list_embeddings`,
`clear_tts_cache`, and `reset_tts_cache` through API v3.7.

    cmake -S plugins/voice_plugin -B .codex-build/voice-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/voice-plugin --config Release --target voice_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -VoicePluginDirectory .codex-build/voice-plugin/plugins

Deploy `.codex-build/voice-plugin/plugins/voice_plugin.dll` into GRIM's `plugins/`
folder. Rebuild the host separately for API v3.7 and removal of its old voice/cache
registrations. Earlier v3.x plugins remain supported by the new host.

The host retains microphone/Whisper context, streaming and its shared default
session state, voice configuration, synthesis/playback, SAPI queries, and cache
lifecycle. The plugin retains command results, parsing, file listing/removal, and
existing confirmation policy. It imports no host C++ or SAPI symbols. All tools
declare VOICE permission; file-based commands also declare FILESYSTEM. Host cache
lifecycle operations check both permissions.

Original behavior remains: voice capture returns an unspoken transcript; TTS test
defaults and status fields are unchanged; device queries and platform errors keep
their previous results. Cache clear/reset keeps the existing paths and direct
`.wav` removal rules. Only reset requests common-phrase regeneration.

**Existing placeholders remain:** `nevermind` does not stop playback, and
`create_embedding` does not send the bridge request. Missing embedding paths can
still produce the existing dispatch exception. Metadata describes these limits.
Loading/unloading the DLL does not start/stop host audio, streaming, or cache state.

Tests cover all eleven commands using controlled capture/audio/native-query/cache
endpoints and private temporary files. They do not record audio, play sound, start
models, or touch the real voice cache. The production native SAPI helpers receive
a separate syntax check using the installed Windows SDK.

## UI plugin

`ui_plugin` exposes generic operations against the host's existing stable panel
keys using API v3.8. It owns tool metadata, JSON argument validation, and result
formatting; panel lifetime, discovery, and UI-thread execution remain in the host.

| Tool | JSON arguments |
|---|---|
| `ui.list` | `{}` |
| `ui.inspect` | `{"panel_id":"settings"}` |
| `ui.show` | `{"panel_id":"settings"}` |
| `ui.hide` | `{"panel_id":"settings"}` |
| `ui.bring_to_front` | `{"panel_id":"console"}` |
| `ui.poll` | `{"request_id":1}` |

The first five tools queue requests and return a positive request ID. Use
`ui.poll` after frame processing until `status` is `completed`, then inspect
`result.success`. Pending tool success means accepted, not completed. The
completed poll consumes its result. IDs belong to a single loaded plugin
instance; unloading/replacing that instance cancels its unprocessed operations.

Build and validate only this plugin with installed dependencies:

```powershell
cmake -S plugins/ui_plugin -B .codex-build/ui-plugin -G "Visual Studio 17 2022" -A x64
cmake --build .codex-build/ui-plugin --config Release --target ui_plugin --parallel 2
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -UIPluginDirectory .codex-build/ui-plugin/plugins
```

Deploy the completed `.codex-build/ui-plugin/plugins/ui_plugin.dll` into the
application's `plugins/` folder after separately rebuilding the host for v3.8.
This plugin requires v3.8; older v3.x binaries remain compatible with the new
host. The new host no longer registers `console`, `toggle_settings`,
`ui.show_surface`, or `ui.hide_surface`. Existing handler functions are retained.
Surface creation/destruction and OSINT remain host tools.

Tests load the real DLL against a controlled host UI queue and check metadata,
validation, permissions, pending/completed/failure results, ownership, request
limits, and cancellation on unload. The separate UI foundation suite exercises
the production host adapter against real panel management without native UI.
No model, renderer, training, package installation, or application startup occurs.

### Remaining host tools

These groups stay in the host until the plugin API can preserve their existing
services and state. Replacing them with approximate plugin implementations would
change behavior.

| Group | Missing boundary for a behavior-preserving migration |
|---|---|
| Input control | Live perception context and its mouse/keyboard interfaces. |
| UI surface lifecycle | `ui.create_surface` and `ui.destroy_surface` still use the host surface registry; OSINT panel handling remains outside this migration. |

The eleven former AI/NLP/settings tools are retired from agent registration,
with their handler logic retained for possible host/developer use. They are no
longer a pending plugin migration set. See
[retained handlers and rationale](../docs/RETIRED_AGENT_TOOLS.md).
