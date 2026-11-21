# Native GRIM tool plugins

The current host requires [plugin ABI v18](../docs/PLUGIN_API.md).

API v15 uses native provider-owned panels. Native UI plugins import host UI
classes and must match the host compiler, CRT and class ABI. See
[native panels](../docs/PLUGIN_API.md#native-provider-panels).
Windows standalone native UI builds require a matching
`-DGRIM_HOST_IMPORT_LIBRARY=<path>`. Host-tree all-target builds link these
plugins against `GRIM_EXPORT` after building the host. Focused native panel
tests use an isolated headless host import library.

GeoSpatial Plugin owns the globe panel and streams Cesium content through the
host viewport and bgfx context. On Windows, deployment places its runtime DLL
dependencies under `deployed_plugins/geospatial_plugin/dependencies/`. The
loader copies them to the instance cache for native loading.

API v3.14 adds guarded hide (`GRIM_UI_HIDE_CURRENT`). Like `ui.show`, `ui.hide`
accepts optional `expected_runtime_id`; World Lab supplies it automatically when
resolving an observed panel. Existing unguarded hide JSON remains supported.

API v3.13 adds guarded panel show and cancellation through the existing UI service.
`ui.show` accepts optional `expected_runtime_id`; `ui.poll` accepts optional `cancel`.
No table or registration layout changes. World Lab uses these to protect observed
panel references and release interrupted requests. See
[manual UI dispatch](../docs/WORLD_LAB.md#manual-ui-resource-discovery-and-dispatch).

Pair each provider library with a `<library-stem>.tools.json` sidecar.
Set `TOOL_MANIFEST` on `grim_configure_plugin` to copy the manifest alongside
the library in the build output and deployment folder. See
[tool manifests](../docs/PLUGIN_API.md#tool-registration).

Existing installed dependencies are required; this build installs nothing.
The output folder is .codex-build/plugins-standalone/plugins by default.
GRIM_PLUGIN_OUTPUT_DIR can override it. Each plugin subdirectory is also a
standalone CMake project.

Normal host builds (including `cmake --build --preset release -j 100`) automatically
deploy completed plugin bundles into `deployed_plugins/` under the GRIM root.
Building just the `GRIM` target also builds and deploys its plugins. Deployment
checks run even when linking is up to date, repairing missing/stale live files.
Windows and macOS host builds use the same plugin catalog as the standalone
build. macOS produces `.dylib` libraries; Windows `.dll` files cannot load there.
World Lab's Live view adds rooms when these plugins publish their providers.
After updating an older UI-only Mac build, reconfigure its existing build tree
and build/deploy the plugins with the Mac's installed dependencies. A standalone
Mac build can use `cmake -S plugins -B build-plugins -DGRIM_DEPLOY_PLUGINS=ON`
(plus the existing toolchain options), then `cmake --build build-plugins`.
This does not build the host. Plugin visibility does not imply identical native
device support: digital screen capture/automation still lacks a macOS backend.
Each immediate child folder is named after its native library stem. On Windows,
`deployed_plugins/voice_plugin/` contains `voice_plugin.dll` and
`voice_plugin.tools.json`. Private runtime DLLs go in its `dependencies/` folder.
Deployment leaves identical files untouched and marks the bundle pending while
it publishes changes.

Standalone builds default to **no deployment**. Opt in with
`-DGRIM_DEPLOY_PLUGINS=ON` and build `deploy_voice_plugin` or the default target.
Use `-DGRIM_PLUGIN_DEPLOY_DIR=<folder>` to publish to a separate destination.
Disable deployment with `-DGRIM_DEPLOY_PLUGINS=OFF`.

For manual deployment, place a complete folder under the watched root or use
`deploy_plugin.cmake` with the completed library and its manifest. Remove the
folder to unload the plugin. The loader reconciles the root about once a second.
Replacements must retain the provider name. A failure after native cutover
leaves that provider absent.

Reconfiguring an existing build upgrades the former default deployment path.
Old flat files can stay in place during cutover. If you have existing memory
plugin data, stop the host before copying `plugins/memory_plugin_data/` into
`deployed_plugins/memory_plugin/`. The plugin opens its data beside the library.

See ../docs/PLUGIN_API.md for API ownership, reload behavior, limitations, and
focused verification.

Deployment regression checks: `python -m unittest discover -s tests -p test_plugin_deployment.py -v`.
They build a tiny synthetic plugin/host dependency fixture, never the GRIM application.

## Filesystem plugin

`filesystem_plugin` supplies `pwd`, `cd`, `ls`, `mkdir`, and `rm`. These commands
are no longer registered by the host. `clean` and `help` come from `utility_plugin`.
The plugin uses the API header, C++ standard library, and existing nlohmann JSON
headers for copied resource publication. It does not call host filesystem services.

Build only this plugin with the existing toolchain:

    cmake -S plugins/filesystem_plugin -B .codex-build/filesystem-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/filesystem-plugin --config Release --target filesystem_plugin --parallel 2

Validate the real DLL through the loader/API/registry in a temporary directory:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -FilesystemPluginDirectory .codex-build/filesystem-plugin/plugins

Deploy the completed library and manifest to `deployed_plugins/filesystem_plugin/`
with a host and plugins built for the same ABI.
The loader discovers all five tools and the working-directory resource. Older
API versions are rejected before initialization.

The plugin starts in the process working directory each time it loads. `cd`
changes only its private working directory, shared by its five tools, and reload
resets that directory. Its `filesystem.directory` resource exposes a copied UTF-8
path under resource ID `working_directory`. Unload removes the resource. For isolated DLL lifecycle tests, run
`tests/run_plugin_shutdown.ps1 -BuildFilesystemFixture`.

Paths may be absolute or relative, may contain spaces,
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

Deploy the completed library and manifest to `deployed_plugins/application_plugin/`.
The compatibility test uses a controlled
native launch sink and does not open applications or browsers.

## Calculator plugin

`calculator_plugin` owns the expression parser and registers the Model tool
`calculate`. Its implementation lives entirely in
`plugins/calculator_plugin/calculator_plugin.cpp`; the former
`Tools/Math/Calculator` library and standalone calculator test were retired.
No host calculator service or forwarding layer is involved.

Pass a raw expression, for example `calculate sqrt(144) + 2^3`. The tool is
informational and requires no confirmation. It supports arithmetic,
parentheses, constants `pi` and `e`, and the mathematical functions documented in
its registry descriptor. Angles are radians and arithmetic uses finite doubles.
Inputs are limited to 4096 bytes and 64 active recursive parsing levels.

Successful structured output contains `success`, `value`, and `answer`. Failures
contain `success: false`, `error`, `error_message`, and `error_offset` (a zero-based
byte offset). Error codes use the `ERR_CALC_` prefix; success uses `ERR_NONE`.

Build and test only this plugin using the existing toolchain:

    cmake -S plugins/calculator_plugin -B .codex-build/calculator-plugin -G "Visual Studio 17 2022" -A x64 -DGRIM_CALCULATOR_TESTS=ON -DGRIM_DEPLOY_PLUGINS=OFF
    cmake --build .codex-build/calculator-plugin --config Release --target calculator_plugin_tests --parallel 2
    ctest --test-dir .codex-build/calculator-plugin -C Release --output-on-failure

The tests load the real DLL and exercise registration, Model invocation through
ToolRegistry, arithmetic, structured errors, input limits, and provider removal.
They do not start GRIM or a model. To publish the built DLL to the watched folder:

    cmake -DPLUGIN_FILE=D:/G.R.I.M/.codex-build/calculator-plugin/plugins/calculator_plugin.dll -DPLUGIN_MANIFEST=D:/G.R.I.M/plugins/calculator_plugin/calculator_plugin.tools.json -DDEPLOY_DIR=D:/G.R.I.M/deployed_plugins -P plugins/deploy_plugin.cmake

## Utility plugin

`utility_plugin` supplies `clean` and `help`. Their original complete result
payloads are preserved: message, success/error code, category, voice, and color.
Both still ignore arguments. `help` retains a static command list, with retired
`ai_backend` and `reloadnlp` entries removed; use `list_tools` for current registrations.
`clean` retains the original response-only handler; this migration adds no UI
clearing operation. Neither tool requires access to host state.

    cmake -S plugins/utility_plugin -B .codex-build/utility-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/utility-plugin --config Release --target utility_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -FilesystemPluginDirectory .codex-build/filesystem-plugin/plugins -UtilityPluginDirectory .codex-build/utility-plugin/plugins

Deploy the completed library and manifest to `deployed_plugins/utility_plugin/`. The host must be rebuilt from the migrated source to remove
its old `clean`/`help` registrations. An older host rejects the colliding batch
without losing its working commands. The compatibility test checks exact results,
policy metadata, old-host collisions, and removal through folder reconciliation.

## Tool inspection plugin

`tool_inspection_plugin` supplies `list_tools`, `tool_info`, and `tool_stats` using
API v3.2's read-only registry snapshots. The original formatting and argument
behavior are preserved, including exact category matching, alias/display-name
lookup, and the existing statistics integer-prefix parsing. The plugin registers
usage, parameter, and example metadata and requires no confirmation.

    cmake -S plugins/tool_inspection_plugin -B .codex-build/tool-inspection-plugin -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/tool-inspection-plugin --config Release --target tool_inspection_plugin --parallel 2
    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -ToolInspectionPluginDirectory .codex-build/tool-inspection-plugin/plugins

Deploy the completed library and manifest to `deployed_plugins/tool_inspection_plugin/`.
The plugin imports no host C++ symbols; snapshots contain metadata, not handlers.
The snapshot codec is shared as `MMO/Core/Descriptors/ToolDescriptorJson.hpp`.

Tests cover complete metadata, alias/display-name lookup, unchanged result fields,
statistics limits and errors, large entries/results, copy isolation, lifecycle
metadata, and folder removal. API v3.2 permits result messages up to 16 MiB so
catalog listings can exceed 64 KiB; other result field limits stay unchanged.
World Lab object population remains a separate integration step.

## Removed Aliases plugin

The redundant Aliases plugin and its `alias list`, `alias info`, and `alias refresh`
tools have been removed. Use the Application plugin's `applications.*` tools for
application discovery and inspection. API v12 removes the host alias cache and services. Explicit user aliases and
legacy user-record compatibility reads belong to the Application plugin.
See [configuration](application_plugin/README.md).
Remove stale deployed `aliases_plugin.dll` files from plugin load directories.

## System information plugin

`system_info_plugin` supplies `system_info` and hardware resources using API v6.
The DLL owns native detection, serialized snapshot state, resource publication,
command formatting, and cleanup. It stages the initial batch during initialization
and refreshes it on each tool invocation through the generic resource API.
It neither calls the host detector nor reads the startup cache. Failed captures
retain last-good data with explicit stale/error state; reload starts a fresh cache.
See [System Information resources](../docs/SYSTEM_INFO_RESOURCES.md).

    cmake -S plugins/system_info_plugin -B .codex-build/system-info-plugin -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=OFF
    cmake --build .codex-build/system-info-plugin --config Release --target system_info_plugin --parallel 2

Deploy only alongside a host and plugins rebuilt for API v6. No deployment occurs
in the tests. The tool is informational.
For deterministic DLL/resource tests, run
`tests/run_plugin_shutdown.ps1 -BuildSystemInfoFixture`. Native detection/linking
can be checked with `tests/run_system_info_native.ps1`.

## Schedule and memory plugins

`memory_plugin` owns staging, storage, and persistence. The orchestrator reads
copied hits through the System tool `memory.retrieve`. Pending and short-term
records survive unload and reload. See [Manual memory](../docs/PLUGIN_API.md#manual-memory).

Build the focused Schedule DLL fixture and run lifecycle checks with:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -BuildSchedulingFixtures

The fixture stays outside the live plugin directory. API v8 requires matching
host and plugin rebuilds before deployment.

## Perception plugin

API v9 moves digital/physical capture, primitive operators, workers and observation
state into the DLL. The legacy host service, context manager, environment panels,
question routing and session/memory projection connections are removed.
See [provider ownership, tools and validation](perception_plugin/README.md).


## Voice plugin

API v10 moves capture, transcription, wake detection, TTS, playback, caches and
bridge processes into the provider. The former host-forwarding command names and
voice ABI services are removed. Loading is idle; explicit tools configure and
start runtime work. See [voice provider documentation](voice_plugin/README.md)
for the tools, breaking application connections and standalone lifecycle tests.

## UI plugin

`ui_plugin` exposes generic operations against the host's existing stable panel
keys using API v3.8. It owns tool metadata, JSON argument validation, and result
formatting. The host executes generic UI operations, but resource lifetime
belongs to each declaring provider. Calendar is owned by Schedule and survives
unloading the UI tools provider.

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

Deploy the completed library and manifest to `deployed_plugins/ui_plugin/`
with a matching host build.
This plugin uses the panel operations introduced in v3.8. The new host no longer registers `console`, `toggle_settings`,
`ui.show_surface`, or `ui.hide_surface`. Existing handler functions are retained.
Surface lifecycle and OSINT findings panel handlers remain in the host.

Tests load the real DLL against a controlled host UI queue and check metadata,
validation, pending/completed/failure results, ownership, request
limits, and cancellation on unload. The separate UI foundation suite exercises
the production host adapter against real panel management without native UI.
No model, renderer, training, package installation, or application startup occurs.

## Input plugin

API v11 moves native input into the provider and removes the host forwarding
slot. Four Debug tools remain: `move_mouse`, `click`, `type`, and `key`.
See [ownership and validation](input_plugin/README.md).

## OSINT plugin

API v11 consolidates commands, collection, cache, jobs and Python bridges under
`osint_plugin`. Host consumers read copied tool results. Seven Ignore tools remain;
`osint_show_ui` remains unregistered. See [ownership and validation](osint_plugin/README.md).

## Scheduling plugin (API v8)

`schedule_plugin` owns `schedule.create`, `schedule.get`, `schedule.list`,
`schedule.update`, `schedule.cancel`, `schedule.status`, durable storage and due
notification handling. It
publishes `schedule.state` resources. It also owns Calendar's controller, drafts,
selection, native widgets and actions as a `UIPanel` subclass. Construction
attaches it during Schedule initialization and the host publishes its
resource under Schedule for World Lab.
No Calendar-specific host class, factory, or UI-plugin declaration remains.
Saved records survive unload; overdue delivery resumes on the next provider poll.
There is no host scheduling service or calendar factory.
See [tool contract](../docs/SCHEDULING_TOOLS.md) and
[ownership](../docs/SCHEDULING_SUBSYSTEM.md).

## Application discovery (API v3.18)

The Application plugin now also supplies applications.list, applications.refresh,
applications.inspect and applications.launch. It declares a host application catalog
at initialization; native discovery publishes resources into its provider room.
Existing open/search retain
their original behavior. Rebuild the host and Application plugin together before
using these services; older hosts reject this newer plugin API version.
See [Application discovery](../docs/APPLICATION_DISCOVERY.md) for backend boundaries,
resource identities, guarded launch arguments, testing and platform limitations.

### Console plugin

`console_plugin` owns the native Console panel (`console`), including its widgets,
input queue, and panel lifetime. It declares no agent tools. Generic panel tools
remain in `ui_plugin`; conversation/session/history services remain in the host.
The plugin uses the exported native C++ UI ABI like `schedule_plugin`, so a
standalone Windows build requires `GRIM_HOST_IMPORT_LIBRARY` from the matching
host build. Both native UI plugins depend on the host import library and are
excluded from the host's reverse deployment dependencies to avoid a build cycle.

No host-factory request creates Console. Removing Console Plugin removes its
panel; notifications still trigger popup activity when the panel is absent.
See [Console lifecycle](../docs/ui/CONSOLE.md#provider-ownership).

### Settings plugin

`settings_plugin` owns the native Settings panel (`settings`) and all ten existing
tabs. Constructor registration attaches it through the provider generation lease;
it has no host factory and declares no agent tools. Generic `ui.*` operations
remain in UI Plugin. Configuration persistence, runtime side effects and input
bindings remain shared host services; the plugin imports their native interface.

The matching host/compiler/CRT/UI ABI and Windows `GRIM_HOST_IMPORT_LIBRARY` are
required, as for Console and Schedule. Unsaved edits are discarded on unload;
unload never saves or applies configuration. Keybind widgets release their
capture on destruction. World Lab maps the panel to Settings Plugin.

Focused native DLL tests: `tests/run_ui_input_foundation.ps1` (configuration and
platform effects stubbed; no application/model runtime started).

### Model Workspace plugin

`model_workspace_plugin` owns Training Control (`training_control`) and preserves
all nine tabs. It imports shared host UI components and existing host model,
configuration, resource, and dataset services. No agent tools are added. Its
native build requires the matching host import library; it is excluded from
reverse host deployment dependencies to avoid a build cycle. See
[Training panel ownership and validation](../docs/ui/TRAINING_PANEL.md#model-workspace-provider-ownership).

### Storage plugin

`storage_plugin` owns Shared Storage (`storage`) and its My Files/Devices tabs.
Its native panel attaches during initialization and detaches before DLL shutdown.
UI Plugin supplies generic operations but neither declares nor owns this panel.
No agent tools are added.

The DLL imports host UI and copied storage-service operations; DeviceCommServer,
its network workers, storage index and transfers remain host-owned. The host binds
the service before loading plugins and unbinds it after their retirement. Missing
service initialization is a logged failure. Reload discards widget drafts but does
not stop transfers, clear devices, or disconnect the host hub connection.

Build with the matching host compiler/CRT/UI ABI and `GRIM_HOST_IMPORT_LIBRARY`
for standalone Windows builds. Like the other native panel plugins, Storage is
excluded from reverse host deployment dependencies. Rebuild the host, Storage and
UI plugins together to remove the old factory declaration.

Focused native DLL validation: `tests/run_ui_input_foundation.ps1`. Network and
storage effects are substituted by the headless fixture; no application, model
runtime or network server is started.

### DataHub plugin

`data_hub_plugin` owns `data_hub`, preserving Home, Sources, HuggingFace,
Structurer, Generator and Curriculum. It owns native widgets, drafts, selections
and display caches, not authoritative dataset or job state. It registers no tools.
Generic `ui.*` operations remain owned by UI Plugin.

Host boundaries:

- `ui/data_hub_services.*`: catalog clients, copied configuration/status,
  source/settings persistence, directory collection, environments and rendering.
- `DataCollection/data_hub_jobs.*`: serialized host HF and structuring lanes,
  bounded transient requests, durable downloads and revision-checked batch writes.
- Existing dataset stream, pipeline and generator services retain their lifetimes.

Downloads and batch structuring survive provider unload/reload. A replacement
reads their latest status; it does not restart them. Search/preview/draft results
have host-issued IDs and are discarded when their panel retires. Draft results
are also rejected if their destination changed while generation was pending.
No job captures plugin code, a widget or a panel pointer.

Source edits persist during live interaction, not candidate construction or
destruction. Successful downloads add source records in the host, never widgets
from a worker thread. Source writes retain newly appended download records and
reject conflicting external edits; Reload Sources explicitly discards local drafts.
Settings and queue persistence use atomic host writes. Unsaved generation/editor
drafts are not preserved across reload.

Native builds require the matching host/compiler/CRT ABI and Windows
`GRIM_HOST_IMPORT_LIBRARY`. Rebuild the host, DataHub and UI plugins together.
Do not compile duplicate dataset, HTTP or job implementations into the DLL.

Focused validation (no application, training, model server or real HTTP):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -DataHub
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_data_hub_jobs.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_dataset_stream.ps1
```
