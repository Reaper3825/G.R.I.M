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
Both still ignore arguments. `help` retains its existing static command list,
including historical command names; use `list_tools` for current registrations.
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

### Remaining host tools

These groups stay in the host until the plugin API can preserve their existing
services and state. Replacing them with approximate plugin implementations would
change behavior.

| Group | Missing boundary for a behavior-preserving migration |
|---|---|
| Memory | The existing memory store/rotation pipeline; API memory operations are stubs. |
| Timers | Host-owned pending timers and expiry delivery; plugin timers are not scheduled by the frame loop. |
| Voice | Active voice configuration, engine/session state, cache and audio services; `list_voices` also reads Coqui configuration. |
| Perception/input | Live perception context and its input-control interfaces. |
| UI | Host-owned panels and surface registry lifetime. |
| AI/NLP/settings | Shared configuration, conversation history, grammar state, and model routing. |
| System information | Full hardware detection; existing API queries expose only OS and CPU text. |
