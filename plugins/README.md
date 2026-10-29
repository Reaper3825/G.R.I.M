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

### Remaining host tools

These groups stay in the host until the plugin API can preserve their existing
services and state. Replacing them with approximate plugin implementations would
change behavior.

| Group | Missing boundary for a behavior-preserving migration |
|---|---|
| Aliases | Shared loaded alias data, metadata, and refresh lifecycle. |
| Memory | The existing memory store/rotation pipeline; API memory operations are stubs. |
| Timers | Host-owned pending timers and expiry delivery; plugin timers are not scheduled by the frame loop. |
| Voice | Active voice configuration, engine/session state, cache and audio services; `list_voices` also reads Coqui configuration. |
| Perception/input | Live perception context and its input-control interfaces. |
| UI | Host-owned panels and surface registry lifetime. |
| AI/NLP/settings | Shared configuration, conversation history, grammar state, and model routing. |
| System information | Full hardware detection; existing API queries expose only OS and CPU text. |
| Tool inspection | Complete registry metadata and statistics discovery through the plugin ABI. |
