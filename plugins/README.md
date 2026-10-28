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
