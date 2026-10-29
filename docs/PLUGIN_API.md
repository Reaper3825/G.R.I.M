# G.R.I.M Plugin API v3

API version: **3.8.0**. Updated October 29, 2026.

The host scans the immediate children of its plugins folder for native
libraries (.dll, .so, or .dylib). A plugin exports grim_plugin_get_info,
grim_plugin_init, and grim_plugin_shutdown.
The loader rejects incompatible major versions and newer unsupported versions.
Existing v1/v2 binaries must be rebuilt and migrated; they are not adapted.

Version 3.1 consumes two of the function table's reserved pointer slots, retaining
the table size and existing v3.0 service-field offsets. Existing v3.0 plugins remain supported
by the new host. A plugin using the new services declares v3.1 and is rejected by
an older host rather than reading unavailable function pointers.

## Tool registration

Version 3.8 adds nonblocking stable-key panel operations for the UI Plugin.
It consumes one reserved function pointer slot, preserving the table size and
existing service offsets. Earlier v3.x plugin binaries remain supported.
See [generic UI operations](#generic-ui-operations-v38).

MMO::ToolRegistry is the authoritative store for metadata and executable
handlers. Plugins call api->register_tools(api, tools, count) with a batch
of GrimToolRegistration values. Each contains a descriptor, handler, and
plugin-owned user_data.

Descriptors require a canonical tool ID, display name, tool version,
description, usage, category, and argument format. They can additionally supply:

- Named parameters with types, descriptions, and required flags.
- Usage examples, aliases, keywords, and capability tags.
- Preconditions, context requirements, and affordance tokens.
- Required permission bits, informational/action classification, and a
  confirmation requirement.
- An argument schema encoded as JSON.

The argument format is raw_text or json. A json tool must supply a JSON
schema object. The host validates that the schema is parseable and object-valued;
it does not implement general JSON Schema validation of calls. Argument
validation remains the responsibility of the caller and handler. Parameter lists
are documentation, not an automatic command-line parser. Existing plugins
explicitly describe their raw command-line argument strings.

The host derives provider name, source path, and a unique loaded-instance ID from the plugin. Plugins
cannot supply another provider's identity, load status, or execution statistics.
Tool permission requirements must be a subset of the plugin's declared
permissions. Metadata flags describe policy; direct registry invocation does
not itself implement confirmation or the orchestration policy gate.

Every supplied pointer is borrowed only for the registration call. The host
copies all strings, parameters, and counted arrays before returning. A plugin
can register metadata from stack-local storage. The host retains the user_data
pointer; the plugin must keep its pointee alive until shutdown after active
calls have drained.

Structures have struct_size; API v3 currently requires the exact size.
Flags use uint32_t with values 0 or 1. Each batch, list, or parameter array is
limited to 256 entries. Strings are NUL-terminated, limited to 65,536 bytes,
and required text must be nonempty. Empty optional lists use {nullptr, 0}.
Duplicate parameter names, IDs, and alias collisions are rejected. An alias is
part of its canonical tool, not a separate tool registration.

### Initialization transaction

The loader creates a distinct host-owned API table per loaded instance. Its host_context
identifies the provider. Pass that API table as self on every call; retain it
through shutdown.

During initialization, registration calls copy and stage batches. No staged
tool is discoverable or executable. Once grim_plugin_init returns GRIM_OK,
the loader validates and publishes all staged tools in one registry batch.
Any failed registration poisons the initialization transaction, even if the
plugin ignores its error. Cross-batch duplicates and collisions are checked
at commit. A failed initialization or commit publishes no partial tool set.

After initialization, each register_tools call is an immediate atomic batch.
Registration, event/timer operations, and lifecycle changes must be serialized
by the host/caller. This API does not make all plugin services thread-safe.
unregister_command removes a committed canonical tool owned by the caller;
is_command_registered queries committed registry entries.

## Result ownership

Handlers have this signature:

    GrimResult handler(const GrimToolCall* call,
                       GrimToolResultWriter* writer,
                       void* user_data);

The host supplies a synchronous call with an argument string and byte length.
The handler completes exactly once with writer->complete(writer, &result).
The writer copies the result's message, error code, category, voice text, and
optional JSON output immediately, while the handler's local storage is alive.
The host owns those copies; neither side frees strings allocated by the other.

GrimToolResult::success reports whether the operation succeeded.
The handler's returned GrimResult reports whether it delivered a valid result.
Missing completion, repeated completion, malformed JSON output, invalid result
structures, or a non-OK handler return produce ERR_PLUGIN_RESULT. Writer
validation failures are remembered even when a handler ignores the error.
Exceptions from handlers become registry command failures.

Result messages support up to 16 MiB as of v3.2 so complete catalog listings can
cross the plugin boundary. As of v3.6 voice text also supports 16 MiB to preserve
complete perception observations. Other result strings retain the 65,536-byte limit.
Optional JSON output must
be valid JSON and is preserved as CommandResult::structuredOutputJson.

The call, writer, and their pointers must not escape the callback or be used
from background threads. A background operation can return a job identifier;
it must expose later status through a separate tool. There is no asynchronous
completion API in v3.

execute_command(api, command_text, writer) also delivers through a synchronous
writer. The receiving plugin copies any strings it needs before its completion
callback returns. There are no returned result strings to free.

### Minimal plugin

    #include "core/plugin_api.hpp"

    static GrimResult echo(const GrimToolCall* call,
                           GrimToolResultWriter* writer, void*) {
        GrimToolResult result;
        result.success = 1;
        result.message = call->arguments;
        result.category = "information";
        return writer->complete(writer, &result);
    }

    extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
        static const GrimPluginInfo info{
            GRIM_API_VERSION, "Echo", "1.0.0", "Example",
            "Demonstrates a complete registered tool", 0
        };
        return &info;
    }

    extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
        if (!api || api->api_version != GRIM_API_VERSION)
            return GRIM_ERROR_API_VERSION_MISMATCH;
        GrimToolParameter text;
        text.name = "text";
        text.type = "string";
        text.description = "Text to return";

        GrimToolRegistration tool;
        tool.handler = echo;
        auto& d = tool.descriptor;
        d.tool_id = d.display_name = "echo";
        d.version = "1.0.0";
        d.description = "Return the supplied text";
        d.usage = "echo [text]";
        d.category = "information";
        d.argument_format = "raw_text";
        d.parameters = &text;
        d.parameter_count = 1;
        d.is_informational = 1;
        return api->register_tools(api, &tool, 1);
    }

    extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}

## Shutdown and native lifetime

Before calling shutdown, the host closes the provider's tool-call gate,
removes its registry entries, and waits for active handlers to finish.
An invocation retaining an old registry entry cannot enter plugin code after
the gate closes. Handler code and user data remain valid through completion.

Shutdown must stop and join plugin-owned background work. It is also called
after failed initialization, so it must tolerate partially initialized state.
After shutdown, host cleanup removes provider-owned subscriptions, timers, and
the API context, then the loader unloads the native library. Do not retain the
API pointer after cleanup.

Lifecycle operations run on the host lifecycle thread, never inside a plugin
callback: waiting for one's own active call would deadlock. Plugin-created
threads must not mutate the API's shared service maps concurrently with cleanup.
Event and timer subscriptions carry the same instance identity and call gate.
Staged callbacks cannot run before publication. Snapshots of pending callbacks
cannot enter retired plugin code after its gate closes. The lifecycle waits for
active tool, event, and timer callbacks before unloading the library.

The migrated OSINT plugin joins outstanding scans at shutdown. The network
plugin joins its coordinator, which owns worker joins and network cleanup.

## Other API services and current limits

The function table retains logging, context, events, timers, output, voice,
filesystem, process, system-information, allocation, and string utilities.
Check individual implementations before depending on a service: several are
partial or stubs. Memory store/retrieve/delete remain stubs, and classify_intent
returns an unmatched result without local classification.

Host-returned strings from other services retain their existing string_free
ownership contract; the new result writer applies specifically to tool results
and command execution. The API does not sandbox native code. Declared
permissions gate supported host services, not arbitrary native OS calls.

## Folder reconciliation and reloads

The application calls PluginManager::tick from the host frame loop. It polls the
configured folder about once per second; explicit API reload requests are queued
and serviced on a subsequent frame. There is no watcher thread executing lifecycle
operations, and request_reload never unloads code from inside its callback.

A complete directory scan reconciles additions, changed files, and deletions.
A missing folder is treated as empty and can appear later. Directory access
errors preserve the currently loaded set. Scanning is non-recursive and ignores
directories and core_plugin libraries. Change detection includes modification
time, size, and a content digest, so a replacement with a preserved timestamp is
still detected. An unchanged rejected binary is not retried on every poll. It is reconsidered
after another plugin changes the loaded surface, allowing a resolved identity or
alias collision to recover without rewriting the rejected file.

Each library is copied into an exclusively created temporary directory before
loading. The source remains unlocked on Windows and can be replaced or deleted.
A source/copy digest check rejects a file that changes during copying. Owned
temporary directories are removed when their instance retires; a process crash
can leave a temporary directory behind. Do not place companion libraries in the
scanned top-level folder unless they are themselves GRIM plugins. Windows loads
can resolve companion dependencies from the source folder; on Unix, dependencies
must be resolvable through the normal loader paths (source-relative $ORIGIN
dependencies are not relocated automatically).

For a changed file, the current plugin keeps serving tools while a separate
candidate library initializes with its own API context and staged registrations.
A replacement must retain its declared provider name. A bad binary, incompatible
ABI, failed initialization, or registry collision rejects only the candidate.
Its subscriptions, timers, and registrations cannot erase the working instance's
resources.

Successful publication atomically replaces the old instance's entire tool set,
including removing tools and aliases no longer supplied. The host then closes
the old instance's callback gate, drains active calls, runs shutdown, removes
only its resources, and unloads its private library copy. An old call's statistics
do not get attributed to its replacement. A successful reload starts a new
instance with fresh statistics.

This transaction protects the registry and host-owned plugin resources. Arbitrary
external changes performed by plugin initialization are not rolled back.
Initialization should register capabilities and allocate reversible local state;
reserve external actions for explicitly invoked tool handlers.

Plugin timers are still not scheduled by the application frame loop; timer
dispatch is available through PluginAPI::processTimers for controlled callers.
This phase wires plugin reconciliation, not a new general timer scheduler.

## Generic UI operations (v3.8)

`ui_panel_service(self, operation, panel_key, request_id)` requires `GRIM_PERM_UI`
and a published, accepting plugin instance. Initialization cannot submit UI work.
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
mean invalid arguments, denied permission, a nonpublished/stopping provider, or
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
Mutations carry confirmation metadata; direct registry invocation does not
enforce that policy. The old `console`, `toggle_settings`, `ui.show_surface`, and
`ui.hide_surface` registrations are removed. `ui.create_surface` and
`ui.destroy_surface` remain host-owned lifecycle tools, and OSINT is unchanged.
See [UI panel operations](UI_PANEL_OPERATIONS.md) for stable keys and examples.

## Independent plugin builds

The native plugins build without linking GRIM, importing host C++ symbols, or
configuring the application. Use the standalone CMake entry point and existing
installed dependencies:

    cmake -S plugins -B .codex-build/plugins-standalone -G "Visual Studio 17 2022" -A x64
    cmake --build .codex-build/plugins-standalone --config Release --target osint_plugin network_test_plugin --parallel 2

On Windows, nlohmann_json is discovered from the existing vcpkg_installed tree.
Other platforms can supply its installed package through CMAKE_PREFIX_PATH.
Nothing downloads or installs packages. Each plugin subdirectory can also be
configured independently with its own build directory.

Libraries are emitted under the build directory's plugins folder. Set
GRIM_PLUGIN_OUTPUT_DIR to an explicit destination if needed. Build outputs are
not automatically deployed into the application's live plugins folder.
Copy a completed library there to install/update it; remove it to unload it.

Logging passes through the plugin API using a private per-library bridge.
The OSINT findings panel stays in the host as ui.show_osint_results; the
osint_show_ui plugin tool delegates to that host tool via execute_command.
This keeps host UI objects and their lifetime out of replaceable plugin DLLs.

## Verification

Run the focused host-only suite with an existing MSVC installation:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1

It compiles tiny plugin DLL fixtures and the real loader/API/registry with host
service sinks. Coverage includes folder loading, ABI rejection, failed-init
cleanup, atomic initialization, aliases, copied metadata/results, malformed or
missing results, provider ownership, permission declarations, active-call draining,
folder additions/removals, queued reloads, rejected replacements, successful
cutovers, and generation-isolated resource cleanup. It builds and runs no GRIM
application or training/runtime target.

To inspect registrations from the independently built real plugins as well,
without invoking scans or network tests:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_plugin_shutdown.ps1 -PluginDirectory .codex-build/plugins-standalone/plugins

The Tools viewer uses the same descriptors, including argument format/schema
and plugin source path. Its focused host UI regression is:

    powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -Tools

### Filesystem migration

`filesystem_plugin` owns the former built-in `pwd`, `cd`, `ls`, `mkdir`, and `rm`
tools. It registers one atomic batch with path parameters, examples, usage,
filesystem permissions, and informational/confirmation metadata. It uses no
host globals or imported host symbols. `clean` and `help` are provided by the
separate `utility_plugin`.

The plugin's working directory starts at the process working directory on each
load; `cd` changes only plugin-local state. Reload resets it. The registry is
populated solely by discovering and initializing the plugin file. Removing that
file removes the five registrations. Older hosts that still register these IDs
reject the plugin batch rather than replacing the built-ins.

See [plugin build and deployment instructions](../plugins/README.md#filesystem-plugin).
The optional `-FilesystemPluginDirectory` argument to `tests/run_plugin_shutdown.ps1`
exercises the built DLL in a private temporary folder, including metadata,
absolute/relative navigation, paths with spaces, failed operations, nonrecursive
removal, and removal of the deployed plugin file. No application or model runs.

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
tests exact result compatibility through the real DLL/API/registry, old-host
registration collisions, and removal of the deployed DLL. See the
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

The existing status-only `launch_app` service retains its success/failure and
permission behavior, using the same native primitive without depending on the
removed `cmdOpenApp` handler. Neither service confirms that an application has
finished opening or that a web search completed.

The focused suite's `-ApplicationPluginDirectory` option loads the real DLL and
checks exact result compatibility, raw/trimmed arguments, native status 32/33,
current host error messages, permissions, and DLL removal. The OS primitive is
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
service offsets. Existing v3.0/v3.1 plugins remain supported. New plugins must
check v3.2 before accessing `get_tool_snapshot(self, tool_id)`.

This service returns a host-owned JSON string, released with `string_free`:

```json
{"format":"grim-tool-snapshot","version":1,"tools":[]}
```

- A null `tool_id` selects all registered metadata in category/ID order. Each
  call copies the registry's current entries under its existing mutex.
- A supplied string uses the registry's existing canonical-ID, display-name, and
  alias lookup. Missing tools yield an empty `tools` array; a null return means
  API failure. Unavailable registrations remain inspectable.
- Each entry includes every field of `ToolDescriptor`: identity and provider
  generation/source, documentation, arguments/schema/parameters, policy,
  prerequisites/context/affordances, aliases/keywords/tags, lifecycle, and stats.
  `provider_type` is numeric (0 built-in, 1 plugin); `swap_state` is numeric
  (0 loaded, 1 loading, 2 unloading, 3 unavailable). The version-1 codec lives in
  `MMO/Core/ToolDescriptorJson.hpp`.
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

### Application alias migration (v3.3)

`aliases_plugin` supplies the existing canonical command IDs `alias list`,
`alias info`, and `alias refresh`. These application aliases are separate from
ToolRegistry command aliases. The host keeps ownership of `aliases.cpp`, loaded
user/auto data, persistence, normalization, lookup, scanning, and refresh lifecycle.
The plugin owns command formatting, metadata, and results. Loading or unloading
the plugin does not initialize, clear, scan, or otherwise change the host store.

API v3.3 consumes two more reserved slots while retaining table size and earlier
service offsets. Existing v3.0/v3.1/v3.2 binaries remain supported. The alias
plugin requires v3.3 before reading either new function pointer.

- `get_alias_snapshot(self, key)` returns host-owned version-1 JSON:
  `{"format":"grim-alias-snapshot","version":1,"entries":[],"info":""}`.
  A null key copies `aliases::getAll()` into ordered `{name,target}` entries,
  preserving the host map's current iteration/merge behavior. A supplied key
  returns `aliases::info(key)` text without plugin-side trimming or normalization.
  Returned strings must be released with `string_free`; null means API failure.
- `refresh_aliases(self)` requires `GRIM_PERM_FILESYSTEM` and calls the existing
  synchronous `aliases::refreshNow()`. It returns host-owned JSON:
  `{"format":"grim-alias-refresh","version":1,"completed":true,"error":""}`.
  `completed` means the host function returned normally, not that a scan happened
  or its returned CommandResult indicated success. A denied permission or caught
  standard exception produces `completed:false` with error text. Null means API
  failure. No registered command is dispatched by this service.

The old command wrappers' behavior is intentionally retained: list/refresh ignore
arguments; an empty info argument fails; a missing name normally succeeds with
the host's nonempty "not found in aliases" text. Only empty host info uses the
configured `ERR_ALIAS_NOT_FOUND` message. Refresh reports its original fixed
success result whenever the host function returns, including its busy or failed
status returns, and preserves the original failure response on standard exceptions.
The service does not repair or replace the existing host refresh implementation.

The optional `-AliasesPluginDirectory` test argument loads the real DLL against
a controlled alias-store/scanner fixture. It checks every result field, host
iteration order and live data, lookup argument handling, permissions, refresh
normal/busy/failed/exception outcomes, large listings, and removal without clearing
host aliases. No live application-directory scan or alias database write occurs.
See [build and deployment instructions](../plugins/README.md#aliases-plugin).

### System information migration (v3.4)

`system_info_plugin` supplies `system_info`, retaining its original output,
ignored arguments, result/error fields, voice, color, GPU fallback behavior,
and CUDA/Metal/ROCm reporting. The host no longer registers the command.

API v3.4 consumes one reserved slot without changing table size or earlier
service offsets. Existing v3.0-v3.3 plugin binaries remain supported. New
plugins must check v3.4 before reading `get_hardware_snapshot(self)`.

This service requires `GRIM_PERM_SYSTEM` and calls the existing `detectHardware()`
on each request, matching the former command rather than substituting the cached
startup inventory. Loading/unloading the plugin performs no detection; requesting
a fresh snapshot does not replace `g_hardwareInventory`. Existing `get_system_info`
behavior remains unchanged.

The service returns host-owned JSON, released with `string_free`:

```json
{"format":"grim-hardware-snapshot","version":1,"hardware":{},"error":null}
```

The `hardware` object carries all HardwareInventory data: OS/architecture,
CPU/RAM, GPU records/count and backend flags, monitor records/count and virtual
desktop geometry, audio devices, Wi-Fi state, voice backend flags, suggested
Whisper model, and `capture_time_ns`. The timestamp is monotonic nanoseconds in
the host steady clock's local epoch, not a calendar timestamp. The codec lives in
`MMO/Core/HardwareInventoryJson.hpp`; no C++ object or detector symbol crosses
the ABI.

Permission denial or API failure returns null. A detector exception produces
`hardware:null` and `error:{kind:"exception",message:"..."}`; a nonstandard
exception uses `kind:"unknown"`. The plugin preserves the original dispatch
exception response, including standard exception text and the unknown-error
fallback.

The optional `-SystemInfoPluginDirectory` test argument exercises the real DLL,
API, registry, and result writer against a controlled hardware detector. Coverage
includes permissions, complete topology and clock round trips, fresh detection
per command, independence from startup state, GPU/no-GPU/unknown-GPU formatting,
both exception paths, and folder removal. It performs no actual hardware probe
or application/model execution. See the
[plugin instructions](../plugins/README.md#system-information-plugin).

### User timer and manual memory migration (v3.5)

`timers_plugin` owns `set_timer`; `memory_plugin` owns `remember`, `recall`, and
`forget`. The host no longer registers these commands. Four v3.5 services consume
reserved slots without changing table size or earlier offsets. Existing v3.0-v3.4
binaries remain supported. The new plugins require v3.5.

Each service returns a host-owned JSON string, released with `string_free`:

```json
{"format":"grim-runtime-service","version":1,"value":null,"error":null}
```

Null pointers indicate invalid input, denied permission, or API failure. Host
standard exceptions are represented by `error:{kind:"exception",message:"..."}`;
nonstandard exceptions use `kind:"unknown"`. Plugins preserve the corresponding
original dispatch error responses.

| Service | Permission | Existing host operation |
|---|---|---|
| `schedule_user_timer(self, seconds, message)` | SYSTEM | Positive seconds; appends a native Timer to the host queue with steady-clock expiry. |
| `stage_manual_memory(self, raw)` | MEMORY | Generates ID/time and stages the original USER/STRING/CONVERSATION record in MemoryBufferRotation, with confidence 0.98, original raw text, `remember ` normalized prefix, and manual/remember tags. |
| `search_memory(self, query)` | MEMORY | Calls the existing storage search with its default result limit/order; returns an array of `{id,raw}` hits. |
| `flush_memory_record_if_present(self, id)` | MEMORY | Checks `getById`, flushes if present, and returns a presence boolean. It does not delete records. |

Command parsing and result behavior remain unchanged. Timers retain `std::stoi`
integer-prefix acceptance and the original invalid/nonpositive errors. Manual
remember parsing still stages the original full text, not a new key/value store.
Recall selects the first existing search hit. Forget preserves its original
search/check/flush loop and success text; metadata now explicitly describes the
absence of deletion. The old `store_memory`/`retrieve_memory`/`delete_memory`
API slots remain stubs and are not used by these commands.

Timer state, expiry result construction, memory rotation, storage, and persistence
remain in the host. DLL unload removes registrations without cancelling pending
user timers or discarding memory. No plugin timer callback or memory worker is
introduced. The separate plugin-callback timer API remains unchanged.

Two existing limitations are retained: `forget` does not erase records, and no
current source caller invokes `checkExpiredTimers()`. The migration does not wire
an expiry scheduler or claim timers are automatically delivered. Direct calls to
the existing expiry checker still produce and consume the original results.

The focused suite accepts `-TimersPluginDirectory` and `-MemoryPluginDirectory`.
It uses the real host timer queue, expiry checker, MemoryBufferRotation,
UnifiedMemoryStorage, and persistence code. Records are written only to a private
temporary database. Coverage includes permissions, validation/results, staging
metadata, delayed storage visibility, search ordering, prefix parsing, unload
survival, exact-once expiry checks, non-deleting forget, and disk reload.
See [timer and memory build instructions](../plugins/README.md#timers-and-memory-plugins).

### Perception observation migration (v3.6)

`perception_plugin` supplies `perception_what_see`, `perception_read_text`, and
`perception_detect_objects`. Input-control commands remain built-in. Capture,
OCR, object detection, model/context handling, and subsystem lifetime stay in
the host; the plugin introduces no capture thread, context cache, OpenCV object,
model initialization, or external C++ symbol dependency.

API v3.6 consumes one reserved slot, keeping table size and earlier offsets.
Existing v3.0-v3.5 plugin binaries remain supported. The new plugin requires
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
completion evidence. To preserve large observations, the result writer now
accepts voice text up to 16 MiB, matching its message limit; other bounds remain
unchanged. No output is silently truncated.

The optional `-PerceptionPluginDirectory` test argument exercises the real DLL,
API, registry, and result writer against controlled host observation services.
It checks permissions, initialization/availability transitions, exact routing and
OCR defaults, ignored arguments, literal/empty/error results, 70-KB text and voice,
exception paths, and DLL removal without changing host state. No screen capture,
OCR engine, object model, or model server runs during validation. The remaining
host input-command source receives a syntax check using existing OpenCV headers.
See [plugin build instructions](../plugins/README.md#perception-plugin).

### Voice and TTS cache migration (v3.7)

`voice_plugin` supplies eleven former built-ins: `voice`, `voice_stream`,
`test_tts`, `list_voices`, `test_sapi`, `tts_device`, `nevermind`,
`create_embedding`, `list_embeddings`, `clear_tts_cache`, and `reset_tts_cache`.
Command parsing, formatting, result fields, and informational/confirmation policy
remain in the plugin. The host owns capture, synthesis, playback, live voice
configuration, default-session history, Whisper context, streaming state, and
cache lifecycle. The plugin imports no host C++ or native SAPI symbols.

API v3.7 consumes one reserved slot without changing table size or earlier
offsets. Existing v3.0-v3.6 binaries remain supported. The new plugin checks v3.7
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
| HAS_STREAM_CONTEXT / START_STREAM | Reads the existing context; starts the existing stream with the default session history and original timer/memory references. |
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
`resources/test.wav` path are preserved. Native plugin filesystem operations are
covered by tool permission metadata; this does not introduce sandboxing.

Existing placeholders are not upgraded by migration: `nevermind` acknowledges
cancellation and checks speaking status but does not stop playback/streaming.
`create_embedding` returns its queued acknowledgment without submitting a bridge
request. Its existing missing-path `substr(npos)` exception behavior is retained.
Metadata identifies these placeholders. DLL load/unload performs no capture,
playback, stream stop, model startup, or cache shutdown.

The optional `-VoicePluginDirectory` test argument uses the real DLL, host service
dispatcher, session context/history, registry, and result writer with controlled
capture/audio/cache/native-query endpoints. Cache and embedding file operations
run only in a private temporary directory. Tests cover all eleven tools, exact
results, shared stream references, live configuration, SAPI status branches,
placeholders, large transcripts, permissions, exception behavior, cache ordering,
and unload survival. Native COM/SAPI code receives a syntax check. Validation
does not record audio, play sound, start a model/server, or modify the real cache.
See [build and deployment instructions](../plugins/README.md#voice-plugin).
