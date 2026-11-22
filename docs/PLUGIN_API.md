# G.R.I.M Plugin API v27.0

API version: **27.0.0**.

Version 27 makes plugin dispatch return the provider's JSON result directly.
`execute_command` returns the registry result as a host-owned JSON string.
Rebuild the host and plugins together. The loader rejects version 26 manifests.

Version 25 removes the color field from `GrimToolResult` and the registry's JSON
result. Rebuild and deploy the host, plugin DLLs, and manifests together.

Version 24 removes the plugin permission bitmask and its API entries. Rebuild
and deploy the host, plugin DLLs, and manifests together.

Version 23 removes the retired event slots from the function table. The loader
rejects v22 binaries. Rebuild and deploy the host, plugin DLLs, and manifests
together.

Version 18 moves provider metadata to the required sidecar manifest. Build and
deploy the host, plugin DLLs, and their matching manifests together.

Version 16 removes legacy intent, synonym and conversation history calls,
response output, memory CRUD stubs, process management, command existence
lookup, formatted logging, unused allocation/string creation helpers, and host
timer callbacks. Plugins can subscribe to system ticks and manage their own
due-work policy. Tool registration now carries serialized canonical
`ToolDescriptor` metadata.
The table layout has changed. The loader rejects v15 DLLs. Rebuild the host
and plugins together before deployment.

Version 14 removes the copied-widget definition/event API. Schedule owns a
native `CalendarPanel : UIPanel`, including its widgets, drawing and callbacks.
Native UI plugins import the host UI classes and must match its compiler, CRT
and UI class ABI. Rebuild the host and plugins together before deployment.
See [native provider panels](#native-provider-panels).

Version 12 removes `get_alias_snapshot` and `refresh_aliases`. User application
aliases and portable/Start Menu discovery now belong exclusively to the application
provider. The old auto cache is ignored; existing user aliases are read without
rewriting the database. See [user alias configuration](../plugins/application_plugin/README.md).
All host/plugin binaries must be rebuilt together for the changed table layout.

Version 11 moves native input execution into `input_plugin` and removes
`input_primitive` and `GrimInputOperation` from the public ABI. OSINT commands,
collectors, child execution and Python bridges are exclusively provider-owned.
The retained question handler consumes `osint_report`; the host findings panel
consumes copied JSON from `osint_show_secrets`. No host collector implementation
or direct panel cache reader remains. See [input](../plugins/input_plugin/README.md)
and [OSINT](../plugins/osint_plugin/README.md). All providers and the host must
be rebuilt together; v10 binaries are rejected.

Version 10 moves the voice runtime into `voice_plugin`. It removes
`voice_service`, `GrimVoiceOperation`, `queue_tts`, `stop_tts`, `is_speaking`, and
`GrimVoiceParams`. New explicit JSON tools and copied status/transcript resources
replace those connections. See [voice ownership and lifecycle](../plugins/voice_plugin/README.md).
All providers require a rebuild for the new table layout.

Version 9 is a breaking perception ownership change. `query_perception`,
`GrimPerceptionQuery`, and raw-command `input_service` are removed from the ABI.
Rebuild the host and every plugin together; v8 binaries are rejected.
There is no legacy forwarding or compatibility adapter.

Perception implementation/state lives in `plugins/perception_plugin`.
The host supplies generic registration, system ticks, resource
publication and logging. New tools return copied JSON observations; no OpenCV
objects, engines, devices, or C++ pointers cross the ABI. See
[perception ownership and tools](../plugins/perception_plugin/README.md).

The input plugin supplies `move_mouse`, `click`, `type`, and `key` using its
own native implementation. Dispatch requires an active provider.
Success reports OS acceptance, not an application's reaction. `click_on` remains
retired; pure targeting geometry and checked-click helpers remain under `core/input`.

Version 5 moved application handling into its plugin and added generic resource
publication. Version 6 moves system information handling into its plugin and
removes host hardware services and unused system/CPU/memory query entry points.
Version 7 removes the host file API and `GrimFileHandle`; filesystem operations
and directory state belong to the filesystem plugin. Version 8 removes host
countdown and scheduling services; both providers own their handling and state.
These are breaking changes:
older DLLs must be rebuilt. Retired API fields are removed entirely; the current
table does not preserve their former offsets. See [application ownership](APPLICATION_DISCOVERY.md)
and [system information ownership](SYSTEM_INFO_RESOURCES.md).

At startup, the host discovers native libraries (.dll, .so, or .dylib) in
per-plugin folders under the deployment root. A plugin exports `grim_plugin_init` and
`grim_plugin_shutdown`. The loader checks the manifest ABI before opening the library.
The loader rejects incompatible major versions and newer unsupported versions.

Version 19 passes a shared host function table and a loaded-instance handle
to `grim_plugin_init(const GrimPluginAPI*, GrimPluginHandle)`. Pass the handle
as the first argument to host services. Rebuild the host and provider DLLs and
deploy their matching manifests together. Desktop startup initializes the
overlay and bgfx before plugin discovery.

## Tool registration

Version 3.8 adds nonblocking stable-key panel operations for the UI Plugin.
It consumes one reserved function pointer slot, preserving the table size and
existing service offsets at the time.
See [generic UI operations](#generic-ui-operations-v38).

Place a UTF-8 `<library-stem>.tools.json` beside each provider library. The
loader expects `schema_version: 2`, the packed integer `api_version` (1769472
for 27.0.0), the provider name and version in `provider` and `version`, and a
`tools` array. A provider with tools exports `grim_plugin_dispatch_tool`.
See [Application's manifest](../plugins/application_plugin/application_plugin.tools.json)
for raw text and JSON tool declarations.

The loader passes the manifest's canonical `tool_id` to the dispatcher. Each
tool declares an `argument_format` of `raw_text` or
`json`. JSON tools provide an `argument_schema` object. `display_name` defaults
to the tool ID, `version` to the manifest version, and `usage_type` to `Model`.

The loader fingerprints the library and sidecar together, so a JSON-only edit
triggers replacement. A missing, unreadable, or invalid manifest leaves a
previously loaded generation serving while its source DLL remains present.

### Host tool execution mode

The shared C++ `ToolExecutionMode` selects the execution/feedback source per
invocation. `ToolRegistry::invoke` accepts optional `ToolInvocationOptions`.
The default mode is `Production`, which calls the registered handler and records
execution statistics. A caller can supply an expected registry revision and
require the canonical ID of a tool classified as `Model`. The registry applies
these checks when it selects the handler.

`Sandbox` is a reserved option. Selecting it currently returns
`ERR_TOOL_SANDBOX_NOT_IMPLEMENTED` without calling a handler or updating its
execution statistics. There is no simulation or fallback to Production. This is
a host invocation option, not a tool-descriptor field, global mode, or plugin ABI
change; individual tool registrations are unchanged.

### Tool usage classification (v3.10)

Each tool has a `ToolUsage usage_type` in the canonical descriptor. Plugin
registrations set `ToolDescriptor::usage_type` using `ToolUsage`:

| Value | Canonical / plugin enum | Intended consumer |
|---|---|---|
| 0 | `Debug` | Developer diagnostics and inspection |
| 1 | `Model` | Model-selectable capability |
| 2 | `System` | Host-controlled operation |
| 3 | `Ignore` | Retained but excluded from model discovery |

The default is **Model**, preserving existing exposure. This field is independent
of the existing `usage` syntax string, category, informational/action flags,
and confirmation policy. Only loaded Model entries appear in
`getModelTools()`, registry model prompts, the RL action list, and capability-gap
matching. Developer inspection, full snapshots, host command discovery, and
explicit dispatch continue to include all classifications. Classification is
discovery metadata, not a new authorization or execution gate.

Perception and voice provider tools are Debug. Their implementations and runtime
state belong to their respective plugins. Legacy voice cancellation and
embedding-submission placeholder tools are removed.
All OSINT and network-test plugin tools, plus the host OSINT results-panel tool,
are Ignore.
All alias and filesystem plugin tools are System.
All input plugin tools are Debug.

Invalid usage enum values reject the whole registration batch.

Version-1 `grim-tool-snapshot` entries add numeric `usage_type` with the values
above. The shared codec accepts older snapshots without it as Model, while
rejecting invalid values. Existing snapshot fields retain their validation.
The Tools inspector and `tool_info` display the classification.

### World binding labels (v3.11)

The canonical `ToolDescriptor::bindings` is a map from a usage placeholder name
to an existing world type. For example, `usage: "open {t}"` has
`bindings: {"t": "t"}`. Usage is the model-facing semantic action template;
bindings are system-facing metadata. These labels do not
resolve entities, parse arguments, dispatch tools, or assert world effects.
Existing argument formats, parameter types, schemas, and handlers retain their
meaning. Usage targets use the shared world vocabulary, such as `{t}`, `{c}`,
and `{o}`, rather than subsystem-specific names such as `{panel_id}` or `{path}`.
`<value>` and `[optional]` continue to document ordinary values/options. Usage
labels are not necessarily literal handler calls: `ui.show {t}` describes the
semantic action, while its argument schema describes the actual JSON
format. World Lab dispatches canonical model tools through the registry using their
declared argument format. See [host tool actions](WORLD_LAB.md#host-tool-actions-in-play).

Supported types are `t` (thing), `o` (object), `c` (container), `s` (supporter),
`r` (room), `d` (door), `k` (key), and `f` (food). Reuse these types by semantic
role across subsystems: a directory is a `c` (container), so its usage labels are
`cd {c}` and `mkdir {c}`. Broad or mixed targets retain `t`; for example,
`rm {t}` can target either a file or a directory. File-only targets use `o`,
as in `create_embedding {t} {o}` and `osint_report {t} [--export {o}]`. Do not introduce
separate types for filesystem or other subsystem-specific representations.
Open targets, filesystem destinations, aliases, tool entries, panel/surface IDs,
click targets, speakers/audio references, and OSINT/network targets carry labels.
Text, search queries, coordinates, durations, request IDs, flags, and tools with
no entity arguments have no bindings. No application/file/person-specific types
are introduced. Classification and model discovery remain unchanged.

Native plugins populate `ToolDescriptor::bindings`, an optional map limited to
256 entries. Names must contain ASCII letters, digits, or underscores and occur
as `{name}` in usage. Invalid bindings reject the registration batch atomically.

Version-1 snapshots add a `bindings` JSON object; missing fields in older snapshots
default to empty. Model prompts and `tool_info` show usage; binding maps remain in
system-facing snapshots and the Tools inspector. No changes to world/asset schemas
or compiled gameplay are included.

### Catalog summaries (v3.12)

`ToolDescriptor::short_description` supplies an authored, single-line summary of
5-12 whitespace-separated words. The host validates it before registration
returns. Nonempty invalid summaries reject the entire batch. Empty
summaries remain valid for legacy and third-party registrations; the catalog
honestly displays `Short description unavailable for this registered tool`.
Existing full descriptions and argument documentation remain separate.

Version-1 snapshots add optional `short_description`, defaulting
to empty when reading older snapshots. The host catalog projects only loaded
Model tools using `getModelTools()` and never invokes their handlers.

### Descriptor metadata

API v3.13 adds `GRIM_UI_SHOW_CURRENT` and `GRIM_UI_CANCEL` to the existing UI
service operation enum without changing table or descriptor layouts. Guarded show
uses the stable panel key and `request_id` as the expected runtime panel identity;
the UI thread checks this before mutation and returns `stale_panel` on mismatch.
Cancel uses a null key and an owned request ID, sets the queued cancellation flag,
and releases the request. It cannot undo effects that have already occurred.
Cross-provider cancellation is rejected like cross-provider polling.

The UI Plugin's `ui.show` JSON accepts optional positive `expected_runtime_id`;
`ui.poll` accepts optional boolean `cancel`. Existing calls retain their meaning.
The no-argument `ui.list` accepts either empty/whitespace input or `{}` when called
directly; other non-object or nonempty arguments remain invalid.
Plugins using these operations require v3.13. World Lab supplies the runtime guard
from its host-owned observed directory; it does not expose that field in the
brief resource projection. See [UI dispatch](WORLD_LAB.md#manual-ui-resource-discovery-and-dispatch).

API v3.14 adds `GRIM_UI_HIDE_CURRENT` with the same guarded identity arguments as
`GRIM_UI_SHOW_CURRENT`. `ui.hide` accepts optional positive `expected_runtime_id`;
World Lab always supplies it from the observed directory. The UI thread rejects
stale generations before hiding. Existing unguarded JSON calls retain their meaning;
table and registration layouts remain unchanged. Guarded hide requires a v3.14 host/plugin.

Descriptors require a canonical tool ID, display name, tool version,
description, usage, category, and argument format. They can additionally supply:

- Named parameters with types, descriptions, and required flags.
- Aliases, keywords, and capability tags.
- Preconditions.
- Informational/action classification and a confirmation requirement.
- An argument schema encoded as JSON.

The argument format is raw_text or json. A json tool must supply a JSON
schema object. The host validates that the schema is parseable and object-valued;
it does not implement general JSON Schema validation of calls. Argument
validation remains the responsibility of the caller and handler. Parameter lists
are documentation, not an automatic command-line parser. Existing plugins
explicitly describe their raw command-line argument strings.

The host derives provider name, source path, and a unique loaded-instance ID from the plugin. Plugins
cannot supply another provider's identity, load status, or execution statistics.
Metadata flags describe policy; direct registry invocation does
not itself implement confirmation or the orchestration policy gate.

The host owns the decoded descriptors and initializes execution statistics.
The manifest can supply up to 256 tool entries. The host checks metadata when reading the manifest and checks duplicate tool IDs and aliases
during publication.

### Provider initialization

The loader stages the provider and its manifest tools before invoking init.
Init can publish resources and attach panels to that staged instance. Tool and
tick dispatch starts once the successfully initialized provider is published.
Failed init goes through the same teardown as unload.

`unregister_command` removes a canonical tool owned by the calling instance.

## Result ownership

A plugin with tools exports:

    extern "C" PLUGIN_EXPORT GrimToolReply grim_plugin_dispatch_tool(
        const char* tool_id, const GrimToolCall* call);

The call and its argument bytes are borrowed during dispatch. The provider builds
and validates its result, then returns UTF-8 JSON bytes, their size, and a release
function in `GrimToolReply`. `grim::toolReply` builds a result using the common
fields and allocates the buffer in the provider. The host parses the bytes and
calls the provider's release function before allowing the DLL to unload. The
returned buffer need not be NUL-terminated.

A delivered result has `status`, `error_code`, `message`, `category`, and an
object-valued `payload`. Providers can add fields. A provider reports
`completed`, `failed`, or `unknown` for the invocation. A successful submission
can carry a pending service status and a job ID in its payload. Later status
is available through the provider's status tool where supported.

`GrimToolReply::delivery` describes whether the dispatcher returned a buffer.
A non-OK delivery code, missing buffer, or unreadable JSON produces
`ERR_PLUGIN_RESULT`. `ToolRegistry::invoke` returns the decoded provider result
and generates its own errors for dispatch failures. World Lab renders the
payload as feedback and records the invocation status on its action receipt.

`execute_command(instance, command_text)` returns the registry result serialized
as a host-owned JSON string. The caller releases it with `string_free`. A null
pointer indicates an API failure.

## Shutdown and native lifetime

The host closes the instance's callback gate and retires its provider record.
Retirement removes the provider, tools, aliases, and resources together. The
host detaches panels and drains active calls, then removes subscriptions and
cancels queued panel requests before invoking shutdown.

Shutdown stops and joins plugin-owned background work. It also runs after
failed init and must tolerate partial initialization. The instance handle
remains valid for teardown logging through shutdown. The loader then
invalidates the handle and unloads the DLL.

Lifecycle operations run on the host frame thread outside plugin callbacks.
Tool calls retain the selected provider instance and use its shared dispatcher.
Tool dispatch, reply release, and tick callbacks use the instance's call gate.

The migrated OSINT plugin joins outstanding scans at shutdown. The network
plugin joins its coordinator, which owns worker joins and network cleanup.

## Other API services and current limits

The function table retains logging, notifications,
manual memory services, and generation-scoped provider resource publication.

Host-returned strings use the `string_free` ownership contract. The API does not sandbox native code.

## Startup and shutdown batches

`PluginManager::loadBatch()` discovers native plugin bundles in sorted order at
host startup. The DLL and manifest live in an immediate child folder named for
the DLL stem. On Windows the loader also searches the bundle's `dependencies`
directory for native imports. The host opens the deployed DLL directly. Stop
the host before replacing a deployed DLL.

Each discovered DLL is initialized in its provider entry. Init can register
resources and subscribe to system ticks using that entry's handle. After
discovery, the registry publishes the successful entries, tools, aliases, and
resources together. Each rejected DLL has a typed outcome and is cleaned up
without blocking the other bundles. ABI v26 removes the reload request field.
Build and deploy all plugins with the updated ABI before restarting the host.

`PluginManager::shutdownBatch()` disables new dispatch and withdraws the batch
from the registry. It drains active calls, retires native panels, and invokes
plugin shutdown before releasing each module in reverse load order.

## System tick subscriptions (v15.2)

`subscribe_system_tick` registers one callback per plugin generation. The host
calls it on the application frame thread. The callback receives elapsed seconds since the previous frame began
and elapsed seconds since the first application frame began. Both values are zero
on the first frame. Subscriptions registered during init start receiving ticks
after publication. Keep the callback context alive through shutdown. The host
drains active callbacks before plugin shutdown and removes the subscription
before unloading the DLL.

## Generic UI operations (v3.8)

`ui_panel_service(self, operation, panel_key, request_id)` requires a published,
accepting plugin instance. Initialization cannot submit UI work.
The service never dispatches another command and never blocks on the UI thread.

- Submit `GRIM_UI_LIST` with a null key and request ID zero.
- Submit `GRIM_UI_INSPECT`, `GRIM_UI_SHOW`, `GRIM_UI_HIDE`, or
  `GRIM_UI_BRING_TO_FRONT` with a nonempty stable key (up to 256 bytes) and ID zero.
- Poll using `GRIM_UI_POLL`, a null key, and the returned positive request ID.

Returned strings are host-owned JSON released with `string_free`:

```json
{"format":"grim-ui-panel-response","version":1,"request_id":1,"status":"pending"}
```

Every accepted submission queues host-owned work and returns `pending`.
`completed` includes `result`, with `success` and either panel metadata, a panel
list, or an error such as `not_found`. A pending response confirms acceptance,
not execution. Panels are addressed by semantic keys, never display titles.
Completion reports the UI state at execution, not a guarantee of future state.

Polling is nonblocking. A completed poll consumes the result; unknown, consumed,
or foreign request IDs return `status:error` with `error:request_not_found`.
Each plugin instance can retain at most 64 requests, including unconsumed
completed results. Further submissions return `too_many_requests`. Null returns
mean invalid arguments, a nonpublished/stopping provider, or
an API failure. Host execution exceptions become failed completed results.

Queued work contains copied host data and cancellation flags, not plugin code,
callback pointers, or API contexts. Shutdown/replacement cancels unprocessed
requests and discards results belonging to the retired instance. Already
executed operations are not rolled back. Requests need an initialized UI and
active frame processing; polling cannot advance the frame loop. No background
UI thread or automatic timeout is introduced. As with existing plugin services,
callers must honor the documented lifecycle/service serialization requirements.

`ui_plugin` registers `ui.list`, `ui.inspect`, `ui.show`, `ui.hide`,
`ui.bring_to_front`, and `ui.poll`, with JSON arguments and complete metadata.
A caller that sets `wait_for_completion` in `GrimToolCall` can
wait for a completed panel operation. The UI plugin polls the native service and
returns its completed result object. Direct callers can still use the asynchronous
request protocol and the system tool `ui.poll`.
Mutations carry confirmation metadata; direct registry invocation does not
enforce that policy. The old `console`, `toggle_settings`, `ui.show_surface`, and
`ui.hide_surface` registrations are removed. Surface creation, destruction, and
OSINT findings panel handlers remain host-owned after their tool registrations were removed.
See [UI panel operations](ui/PANEL_OPERATIONS.md) for stable keys and examples.

## Input tool migration (v3.9) â€” removed in v9

The following describes the retired ABI. It is not available in v9.

`input_plugin` supplies `move_mouse`, `click`, `type`, `key`, and `click_on`.
They are no longer registered by the host. This section describes the retained legacy `input_service` boundary, still used
by `click_on`. For the four direct tools, see the v8.1 primitive ownership
contract above. Legacy handlers and live perception/input services remain host-owned.

API v3.9 consumes one reserved pointer slot, retaining table size and previous
field offsets at the time. The
input plugin requires v3.9 before accessing `input_service`.

`input_service(self, operation, arguments, arguments_size)` requires the new
`GRIM_PERM_INPUT` permission (bit 7). `GRIM_INPUT_CLICK_ON` additionally requires
`GRIM_PERM_SYSTEM` for capture/targeting. UI permission alone does not authorize
input. The host requires a published, accepting provider; initialization cannot
execute input. It copies up to 65,536 argument bytes before calling its existing
host operation. A null argument pointer is valid only for zero bytes.

The service calls `ExecuteInputCommandService` directly, not the registry or
command dispatcher, so a plugin registration cannot recurse into itself.
Execution remains synchronous, including existing typing/key delays and fresh
`click_on` observation. No background scheduler, job IDs, or UI-style polling is
added in this behavior-preserving migration. Active-call gates keep the API and
plugin loaded until an in-flight call finishes.

The returned host-owned string, released with `string_free`, is versioned JSON:

```json
{"format":"grim-input-result","version":1,"result":{"success":true,"message":"...","error_code":"ERR_NONE","category":"routine","voice":"...","structured_output_json":"","color_rgb":65535}}
```

The plugin preserves every result field, including `click_on`'s structured
coordinates and `application_response_verified:false`. Standard/unknown host
exceptions are encoded in an `exception` field and rethrown on the plugin side,
preserving the registry's `ERR_CMD_EXCEPTION` policy. Null denotes invalid API
arguments, a nonpublished/stopping provider, or API failure;
the plugin reports `ERR_INPUT_SERVICE` in that case.

Raw argument behavior is unchanged: coordinate commas/integer-prefix parsing,
default mouse button, verbatim typing with a 15 ms delay, and plus-separated key
handling. The four direct input commands retain their existing void host input
interfaces and success reporting; this migration does not add OS-delivery checks
to them, validate all key/button names, or change platform stubs. `click_on`
retains its separately implemented checked targeting/delivery contract. See
[click targeting](input/TOOLS.md#grounded-dispatch).

All five tools are state-changing and carry confirmation metadata. As elsewhere,
direct registry calls do not enforce that metadata, and native plugin code is
not sandboxed. The permission controls access to this host service.

## Independent plugin builds

The native plugins build without linking GRIM, importing host C++ symbols, or
configuring the application. Use the standalone CMake entry point and existing
installed dependencies:

    cmake -S plugins -B .codex-build/plugins-standalone -G "Visual Studio 17 2022" -A x64 -DGRIM_DEPLOY_PLUGINS=ON -DGRIM_PLUGIN_DEPLOY_DIR=.codex-build/plugins-standalone/deployed_plugins
    cmake --build .codex-build/plugins-standalone --config Release --target deploy_osint_plugin deploy_network_test_plugin --parallel 2

On Windows, nlohmann_json is discovered from the existing vcpkg_installed tree.
Other platforms can supply its installed package through CMAKE_PREFIX_PATH.
Nothing downloads or installs packages. Each plugin subdirectory can also be
configured independently with its own build directory.

Build outputs remain under the build directory's plugins folder by default.
Deployment uses a separate root, deployed_plugins/ under the GRIM root.
Host builds enable deployment by default. Standalone builds can opt in with
-DGRIM_DEPLOY_PLUGINS=ON and build the default target or deploy_<plugin-target>.
Set GRIM_PLUGIN_DEPLOY_DIR to publish to another root for tests or packaging.

The published folder contains the library, manifest, and selected private runtime
DLLs. Identical files keep their timestamps. Stop the host before updating a
deployed bundle. Remove the folder
before the next startup to disable the plugin.
See [build instructions](../plugins/README.md).

Logging passes through the plugin API using a private per-library bridge.
## Verification

Run the focused host-only suite with an existing MSVC installation:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1

It compiles tiny plugin DLL fixtures and the real loader/API/registry with host
service sinks. Coverage includes folder loading, ABI rejection, failed-init
cleanup, atomic initialization, aliases, copied metadata/results, malformed or
missing results, provider ownership, active-call draining,
folder additions/removals, queued reloads, rejected replacements, successful
cutovers, and generation-isolated resource cleanup. It builds and runs no GRIM
application or training/runtime target.

To inspect registrations from the independently built real plugins as well,
without invoking scans or network tests:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -PluginDirectory .codex-build/plugins-standalone/deployed_plugins

The Tools viewer uses the same descriptors, including argument format/schema
and plugin source path. Its Usage type and Plugin dropdowns combine with search;
plugin choices are derived from current registry providers and refresh when
plugins register or unload.

### Filesystem migration

`filesystem_plugin` owns the `ls`, `mkdir`, and `rm`
tools. It registers one atomic batch with path parameters, usage,
and informational/confirmation metadata. It uses no
host globals or imported host symbols. `clean` and `help` are provided by the
separate `utility_plugin`.

On load, the plugin captures the process working directory as its base for
relative paths. It publishes that path in the `working_directory` resource of
kind `filesystem.directory`, with UTF-8 path and provider scope in its state.
Reload captures the base directory for the new generation. A failure after
cutover leaves the provider absent. Removing the deployed folder withdraws
the tools and resource.

The v7 API contains no host file-open/close/read/write/existence/deletion service
or file-handle type. These tools operate directly through the plugin's C++
filesystem implementation; they retain no open file handles between calls.
No additional file-reading or file-writing tools are introduced.

See [plugin build and deployment instructions](../plugins/README.md#filesystem-plugin).
The optional `-FilesystemPluginDirectory` argument to `tests/run_plugin_shutdown.ps1`
exercises the built DLL in a private temporary folder, including metadata,
absolute/relative navigation, paths with spaces, failed operations, nonrecursive
removal, resource updates, failed replacement, reload, and plugin folder removal. Use
`-BuildFilesystemFixture` to build an isolated fixture first. No application or model runs.

### Native calculator plugin

`calculator_plugin` owns its parser and evaluation handler, and registers the
informational Model tool `calculate` in its sidecar manifest. It has no confirmation or host-service dependency. The raw expression parameter
is documented as `calculate {t}` with the text binding `t`.

The result includes a human-readable message and a payload containing
`success`, `value`, and `answer`. Evaluation errors use an `ERR_CALC_` code and JSON
fields `success: false`, `error`, `error_message`, and `error_offset` (a zero-based
byte offset). The plugin bounds input length and recursive parsing, rejects
non-finite results, and keeps C++ parser exceptions within the plugin boundary.

See the [calculator plugin instructions](../plugins/README.md#calculator-plugin)
for the standalone build and DLL ABI and manifest tests.

### Utility migration and compatibility boundary

`utility_plugin` supplies `clean` and `help` with their previous complete result
payloads and registration policy. Both still ignore arguments. This is a migration
of the existing response-only `clean` and static `help` handlers, not a redesign
of console clearing or help discovery. There are no host-state dependencies.

The later retirement of AI/NLP/admin agent registrations removes `ai_backend`
and the historical `reloadnlp` entry from static help. The eleven retained
handlers and their possible host uses are documented in
[Retired agent tools](RETIRED_AGENT_TOOLS.md); they are not pending plugin migrations.

The optional `-UtilityPluginDirectory` argument to `tests/run_plugin_shutdown.ps1`
tests exact result compatibility through the real DLL/API/registry and removal
of the deployed plugin folder. See the
[utility plugin instructions and remaining host boundaries](../plugins/README.md#utility-plugin).
Tools requiring live host state remain built-in until the API can preserve their
full behavior; stub or partial services are not used as migration substitutes.

### Application/browser migration (v3.1)

`application_plugin` owns `open` and `search`. The plugin preserves the former
handlers' argument handling, Google URL construction, success/failure messages,
error codes, categories, voice, colors, and informational/confirmation flags.
`open` keeps raw arguments; `search` trims surrounding whitespace. Non-Windows
stub behavior remains unchanged. The host no longer registers either tool.

Two v3.1 host services provide the required shared boundary:

- `get_user_error_message(self, code)` reads the current host ErrorManager. The
  returned string is copied by the host and must be released with `string_free`.
  Null arguments or lookup/allocation errors return null.
- `open_target(self, target, native_status)` requires `GRIM_PERM_PROCESS`. On
  Windows it uses the same `ShellExecuteA(..., "open", ..., SW_SHOWNORMAL)` call
  as the original handlers, returning its unmodified native status. `GRIM_OK`
  means a status was obtained; a status above 32 means launch success. Other
  platforms return `GRIM_ERROR_UNSUPPORTED`. This service does not dispatch a
  registered command, so it cannot recurse into `open` or `search`.

The existing status-only `launch_app` service uses the same native primitive without depending on the
removed `cmdOpenApp` handler. Neither service confirms that an application has
finished opening or that a web search completed.

The focused suite's `-ApplicationPluginDirectory` option loads the real DLL and
checks exact result compatibility, raw/trimmed arguments, native status 32/33,
current host error messages and DLL removal. The OS primitive is
replaced with a test sink; no applications or browsers are opened. The production
OS implementation receives a separate syntax check. See the
[build and deployment instructions](../plugins/README.md#application-plugin).

### Tool inspection migration (v3.2)

`tool_inspection_plugin` supplies `list_tools`, `tool_info`, and `tool_stats`.
The host no longer registers these tools. The original command bodies retain
their formatting, category filtering, alias/display-name lookup, statistics,
integer-prefix parsing, error codes, voice, and colors. `tool_stats` still
defaults to 20 rows, accepts zero/negative limits as an empty listing, and lets
invalid/out-of-range integer arguments become `ERR_CMD_EXCEPTION` at dispatch.
No new trimming, sorting, or validation policy is introduced.

API v3.2 consumes one additional reserved slot, preserving table size and earlier
service offsets. At the time, existing v3.0/v3.1 plugins remained supported. Plugins using the service had to
check v3.2 before accessing `get_tool_snapshot(self, tool_id)`.

This service returns a host-owned JSON string, released with `string_free`:

```json
{"format":"grim-tool-snapshot","version":1,"tools":[]}
```

- A null `tool_id` selects all registered metadata in category/ID order. Each
  call copies the registry's current entries under its existing mutex.
- A supplied string uses the registry's existing canonical-ID, display-name, and
  alias lookup. Missing tools yield an empty `tools` array; a null return means
  API failure. Plugin tools appear while their provider generation is published.
- Each entry combines tool metadata with the owning provider generation and source.
  Tool metadata includes identity, documentation, arguments/schema/parameters, policy,
  preconditions, aliases/keywords/tags, and stats. Version-1 snapshots emit
  `examples`, `context_requirements`, and `affordance_tokens` as empty arrays for older readers.
  `provider_type` is numeric (0 built-in, 1 plugin). The legacy `swap_state` key
  is emitted as 0 (loaded) for version-1 readers. The codec lives in
  `MMO/Core/Descriptors/ToolDescriptorJson.hpp`.
- Snapshots carry no handlers and do not execute tools or change counters.
  Calling an inspection command still records that command's own usage after
  its handler completes, as before. Snapshot strings remain independent of
  later registrations, metadata changes, and removals.
- Snapshot strings are not restricted to an individual field's 64-KiB limit.
  Command result messages support 16 MiB; other result and registration field
  limits remain unchanged. Oversized command results fail explicitly rather
  than being silently truncated.

The snapshot API provides read-only data for future World Lab catalog population;
this migration does not bind or modify world objects. The optional
`-ToolInspectionPluginDirectory` test argument exercises the real DLL, including
rich snapshots, alias lookup, preserved results, statistics parsing, 70-KB
entries, metadata updates/removals, and DLL removal. See the
[plugin instructions](../plugins/README.md#tool-inspection-plugin).

### Application aliases (v12)

The application provider owns user alias configuration and reads the legacy
FlatBuffer user vector without importing automatic records or writing that file.
Host alias APIs, startup loading, scanning, and cached state are removed.
`open` resolves explicit user aliases; `applications.refresh` reloads configuration
and native discovery. ToolRegistry command aliases are unchanged.
See [provider configuration and compatibility](../plugins/application_plugin/README.md).

### Application catalog descriptors (v21)

Application catalog results return plain resource descriptors. The host assigns
provider identity when publishing resources through `replace_resources`.
API v21 removed the provider identity callback. API v22 removes the
current-provider callback. The host gate blocks new dispatch during retirement
and lets admitted calls drain before shutdown. Rebuild plugin DLLs against the current API.

### System information provider (v6)

`system_info_plugin` supplies the `system_info` tool and its hardware resources.
Its DLL owns detection, last-good snapshot state, resource construction, command
formatting and cleanup. It stages resources with tools during initialization and
publishes subsequent batches through the generic `replace_resources` API.
There is no `get_hardware_snapshot` or `register_hardware_resource` host service.
The historical v3.4 forwarding contract and v3.19 opt-in contract are removed.
See [System Information resources](SYSTEM_INFO_RESOURCES.md) for schemas,
failure handling, native detection and lifecycle validation.

### Manual memory

`memory_plugin` owns memory staging, search, and persistence.
`remember` queues a record and `recall` returns the first stored match. The next
`memory.retrieve` request promotes pending records before performing text and
tag lookup. The orchestrator dispatches this System tool and receives copied
hits and retrieval diagnostics in the result payload.

The provider opens `memory_plugin_data/records.fb` beside the deployed DLL.
It writes pending and short-term records to an atomic transient snapshot at
unload and restores their tiers on load. The storage worker drains and joins
before unload. API v20 removes the three former host memory callback slots. Plugins must be rebuilt for the new table layout.

### Perception observation migration (v3.6)

`perception_plugin` supplies `perception_what_see`, `perception_read_text`, and
`perception_detect_objects`. Input-control commands remain built-in. Capture,
OCR, object detection, model/context handling, and subsystem lifetime stay in
the host; the plugin introduces no capture thread, context cache, OpenCV object,
model initialization, or external C++ symbol dependency.

API v3.6 consumes one reserved slot, keeping table size and earlier offsets.
At the time, existing v3.0-v3.5 plugin binaries remained supported. The new plugin required
v3.6 before accessing `query_perception(self, query)`.

This service requires SYSTEM permission and accepts the typed query values
`GRIM_PERCEPTION_ANALYZE_SCREEN`, `GRIM_PERCEPTION_READ_TEXT`, and
`GRIM_PERCEPTION_DETECT_OBJECTS`. Each call checks the existing host
`Perception::isAvailable()` once. If available, it invokes the corresponding
existing host operation once; OCR uses its original default region arguments
(-1 for x/y/width/height). It returns the version-1 runtime-service envelope
with `value:{available:bool,text:string}`. Host-owned strings are released with
`string_free`. Invalid queries, permission denial, or API failure return null;
standard and unknown host exceptions use the existing runtime-service error
envelope and retain their original dispatch error responses.

All three commands continue to ignore arguments. Unavailable perception yields
the original `ERR_PERCEPTION_NOT_INITIALIZED` result. Available observations
retain their exact text in both message and voice, with `success:true`,
`ERR_NONE`, routine category, and Cyan color, even if text is empty or looks like
an error. The migration does not reinterpret returned prose as readiness or
completion evidence. The plugin still fills the ABI voice slot, which the host
ignores. The host copies the observation in the message field, subject to its
16 MiB limit.

The optional `-PerceptionPluginDirectory` test argument exercises the real DLL,
API, registry, and JSON results against controlled host observation services.
It checks initialization/availability transitions, exact routing and
OCR defaults, ignored arguments, literal/empty/error results, 70-KB text and voice,
exception paths, and DLL removal without changing host state. No screen capture,
OCR engine, object model, or model server runs during validation. The remaining
host input-command source receives a syntax check using existing OpenCV headers.
See [plugin build instructions](../plugins/README.md#perception-plugin).

### Historical voice forwarding migration (v3.7; removed in v10)

The following describes the retired boundary, not the current API.

`voice_plugin` supplies eleven former built-ins: `voice`, `voice_stream`,
`test_tts`, `list_voices`, `test_sapi`, `tts_device`, `nevermind`,
`create_embedding`, `list_embeddings`, `clear_tts_cache`, and `reset_tts_cache`.
Command parsing, formatting, result fields, and informational/confirmation policy
remain in the plugin. The host owns capture, synthesis, playback, live voice
configuration, default-session history, Whisper context, streaming state, and
cache lifecycle. The plugin imports no host C++ or native SAPI symbols.

API v3.7 consumes one reserved slot without changing table size or earlier
offsets. At the time, existing v3.0-v3.6 binaries remained supported. The new plugin checked v3.7
before accessing `voice_service(self, operation, arguments_json)`.

`GrimVoiceOperation` defines typed domain operations; this service does not
dispatch commands. All operations require VOICE permission. Cache shutdown,
initialization, and common-phrase precaching additionally require FILESYSTEM.
Arguments are an optional JSON object (null means `{}`); unknown operations,
malformed/nonobject JSON, permission denial, and API failure return null. Responses
use the existing version-1 `grim-runtime-service` envelope, with standard and
unknown host exceptions preserved. Host-owned strings use `string_free`.

| Operation group | Request/result and existing host behavior |
|---|---|
| CAPTURE | Calls `captureAndTranscribeSpeech(aiConfig)` with the live mutable configuration; returns transcript text. |
| HAS_STREAM_CONTEXT / START_STREAM | Reads the existing context; starts the existing stream with the default session history and original memory reference. |
| XTTS_ENABLED / LANGUAGE / CONFIG | Reads existing live voice state. CONFIG uses the original `aiConfig["voice"]` access semantics. |
| SYNTHESIZE_COQUI | `{text,speaker,speed}` returns a WAV path; retains the original default pitch. |
| PLAY_AUDIO / PLAY_WAV | `{path}` calls the original playback function; PLAY_WAV returns its success boolean. |
| LIST_SAPI / SAPI_DEVICE | Native host COM/SAPI queries return status/description data for the plugin's original formatting. |
| IS_SPEAKING | Reads current host speaking status; does not stop playback. |
| CACHE_SHUTDOWN / CACHE_INIT / PRECACHE_COMMON | Calls the existing cache/voice lifecycle functions, in the order selected by the command. |
| PATHS | Returns the original test-WAV, embedding, temporary/cache, and index paths; no relocation/configuration redesign. |

The command behavior is intentionally retained: `voice` returns the transcript
without invoking reasoning or speaking it again; `test_tts` keeps its original
default phrase and empty category/error-code/voice fields. SAPI COM/enumeration
failures, output-token failures, and successful empty device descriptions retain
their original results. Non-Windows unsupported responses remain unchanged.
`test_sapi` still plays the existing test WAV rather than synthesizing SAPI speech.

Cache commands retain their original direct-directory `.wav` removal rules,
index handling, shutdown/init order, exception responses, and reset precaching.
The original absolute `D:/G.R.I.M/resources/...` cache/embedding paths and relative
`resources/test.wav` path are preserved.

Existing placeholders are not upgraded by migration: `nevermind` acknowledges
cancellation and checks speaking status but does not stop playback/streaming.
`create_embedding` returns its queued acknowledgment without submitting a bridge
request. Its existing missing-path `substr(npos)` exception behavior is retained.
Metadata identifies these placeholders. DLL load/unload performs no capture,
playback, stream stop, model startup, or cache shutdown.

The optional `-VoicePluginDirectory` test argument uses the real DLL, host service
dispatcher, session context/history, registry, and JSON results with controlled
capture/audio/cache/native-query endpoints. Cache and embedding file operations
run only in a private temporary directory. Tests cover all eleven tools, exact
results, shared stream references, live configuration, SAPI status branches,
placeholders, large transcripts, exception behavior, cache ordering,
and unload survival. Native COM/SAPI code receives a syntax check. Validation
does not record audio, play sound, start a model/server, or modify the real cache.
See [build and deployment instructions](../plugins/README.md#voice-plugin).

## Scheduling provider (v8)

`schedule_plugin` owns all six `schedule.*` handlers, durable storage, revision
checks, due-item selection and delivery state. It publishes `schedule.state`
resources (`items` and `delivery`) through generic resource registration.
`items` contains at most 200 records plus the total; tools provide paginated access.
Calendar is a native panel inside Schedule and uses that provider's existing
service. See [native provider panels](#native-provider-panels).

API v8 removes `schedule_user_timer`, `create_local_calendar` and
`schedule_service` outright. Host and plugins must be rebuilt together.
The host dispatches system ticks, gates callbacks by plugin generation, and
delivers copied notifications. Schedule decides which notifications are due.
Saved schedules survive unload, and delivery resumes
on the next provider poll after reload. See [scheduling contract](SCHEDULING_TOOLS.md).

## Native provider panels

On the UI thread, construct the complete panel with host auto-registration
disabled, then call `panel.attachToProvider(instance)` during provider init.
Attachment checks the live provider instance and validates the stable key
against the live UI. The host binds the panel's `ui.panel` resource as
`panel:<stable-key>` under that instance. The provider owns the panel object.

Retirement detaches the panel synchronously before shutdown destroys it.
Detachment checks the panel's runtime identity so cleanup applies to the exact
attachment. Native UI plugins use the host's exported C++ UI interface and
must match its compiler, CRT, and class layout.

Windows standalone builds require `GRIM_HOST_IMPORT_LIBRARY`; host-tree builds link
`GRIM_EXPORT`. Building the host target alone does not build Schedule, since
Schedule imports the host. An all-target build builds/deploys it after the host.

Calendar uses native buttons, input boxes, scrolling and overlay drawing. It
refreshes visible, non-minimized content every 500 ms from `update()`. Storage
operations remain synchronous. Initialization does not open calendar storage;
saved events survive unload, while unsaved widget drafts do not. The generic UI
plugin supplies panel discovery and interaction.

GeoSpatial Plugin owns the `geospatial` panel, globe camera, native viewport
attachment, and Cesium runtime. It connects the attachment during init and subscribes for ticks once per
application frame through the provider call gate. The plugin submits its clear
and scene to the same host bgfx view. Native render callbacks execute on the
application frame thread, which also retires the plugin. Shutdown unregisters
the render callbacks, clears native input, and drains Cesium destruction before
the loader releases the library.

`tests/run_ui_input_foundation.ps1` tests native controls, direct provider attachment,
collision/replacement, and synchronous retirement. It also builds
an isolated Schedule DLL against the headless test host's exports and tests
native draw dispatch, DLL replacement, synchronous detach, and failed init.
No application, renderer backend, model or training process is started.

## Application provider (v5+)

Application discovery, catalog state, and launch handling live in the application
plugin. The old `register_application_catalog` and `application_service` host
services are removed. Initial tools and resources commit atomically; subsequent
updates use generic resource publication. See [Application discovery](APPLICATION_DISCOVERY.md).

## Hardware resources (v6)

Hardware resources use generic publication. The plugin owns its snapshots and
refresh behavior. See [System Information resources](SYSTEM_INFO_RESOURCES.md).

Console is now a native provider-owned panel in `console_plugin` (provider name
`Console Plugin`, stable key `console`), using the same initialization lease as
Calendar. It imports host UI, command, and session services. See
[Console ownership](ui/CONSOLE.md#provider-ownership).

Storage Plugin owns the native Shared Storage panel (`storage`), including My Files
and Devices. It imports the host UI and `ui/storage_panel_services.*`; it registers
no tools and never owns or stops DeviceCommServer. Network bootstrap binds that
service before plugin initialization, and shutdown unbinds it after provider
retirement. Snapshots cross the boundary by value. Panel-local drafts are discarded
on reload; host connection, device, storage and transfer state survives. The native
DLL tests in `tests/run_ui_input_foundation.ps1` exercise both tabs, device actions,
explicit port validation, resource ownership, replacement and failed initialization
without starting networking. Matching host/plugin binaries must be rebuilt together.

DataHub Plugin owns `data_hub` and all six tabs. It imports native UI and host
catalog/service APIs; no domain runtime is compiled into the plugin. The host
starts `DataHubJobs` with dataset services and drains it at application shutdown.
HuggingFace downloads and batch structuring continue through provider retirement.
Search, preview and draft-generation request IDs are released on panel destruction;
host workers never retain provider callbacks or widgets. A replacement reconnects
to copied durable status. Batch writes require the captured catalog revision.

Source/configuration persistence is host-owned and atomic. Candidate construction
and destruction do not save panel state. Live source edits are saved through an
optimistic merge that retains newly downloaded sources; conflicting external edits
are reported and require Reload Sources. Settings save after live interaction.
Canonical rendering and concept supervision semantics are unchanged.
