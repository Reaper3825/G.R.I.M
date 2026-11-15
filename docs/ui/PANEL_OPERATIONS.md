# Stable panel identity and host operation foundation

## Automatic native panel registration

`UIPanel` registers its address with `UIRoot` during construction and unregisters
that exact instance during destruction. `m_panels` and the stable-key index are
non-owning; a provider may own a native panel with a member or `unique_ptr`.
Copying/moving panels is disabled. Duplicate stable keys throw during construction;
a derived constructor failure unwinds registration through the base destructor.

Native panel construction, use and destruction must occur on the UI thread.
Constructors must not reenter the UI pump: registration occurs before derived
initialization finishes. Worker-thread construction is rejected. Lookup results
are borrowed pointers and must not survive panel destruction. Worker callers use
the queued operations returning copied state instead. Owners must destroy native
panels on the UI thread before unloading their implementation DLL.

`addPanel(shared_ptr<UIPanel>)` remains a compatibility adoption operation for
existing host factories and dynamic surfaces. It retains
ownership separately and does not duplicate constructor registration. Removal
releases this retained ownership after detachment. During provider initialization
the loader supplies the provider-generation scope.
Constructor registrations are staged, validated and attached after commit.
Native provider retirement detaches synchronously before DLL shutdown/unload.
Calendar now follows this path and owns its native widgets directly.

## Provider-defined panel ownership contract

This is the required boundary for the provider-defined panel migration. It is
not a claim that the current factory-key API already supports provider-defined
content. The remaining sections describe the current implementation.

### Owners and responsibilities

| Owner | Responsibility |
|---|---|
| Individual provider | Registers its own panels; owns their definitions, content, panel-specific state, actions, and lifecycle. |
| UI provider | Owns its provider state and generic UI discovery/interaction tools. Operating on another provider's panel does not confer ownership of that panel or its domain state. |
| `UIRoot` | Borrows provider-owned native panels, owns remaining host factory instances, and executes attachment, layout, rendering, focus, input routing, and detachment on the UI thread. It implements shared UI mechanics, not provider-specific feature logic. |
| `ProviderRegistry` | Records authoritative provider-generation ownership and copied panel resources. |
| World Lab | Projects registered ownership into provider rooms and consumes discovered resources. It neither assigns nor transfers panel ownership. |

The panel's owning provider and the provider submitting a UI operation are
distinct identities. The latter owns that operation's request/result, not its
target. A panel rendered by the host remains a resource of its declaring
provider. The UI provider registers only panels that actually belong to it,
not a central list of every provider's panels.

### Identity and state

- Ownership MUST be derived from the registering provider's API context and
  bound to its loaded generation; a supplied provider name cannot assign it.
- Every live panel MUST have an unambiguous stable identity and a distinct
  runtime identity. Titles are presentation, not identity. Conflicting
  registrations MUST fail explicitly rather than replace another panel.
- Panel resources MUST be published under the declaring provider's generation,
  including when host infrastructure publishes copied visual state on its behalf.
- Providers own content and domain state. `UIRoot` is authoritative for actual
  visual state such as geometry, visibility, focus, and ordering. UI tools and
  World Lab consume snapshots; neither maintains a second authoritative copy.
- Provider-specific actions MUST route to the owning provider. Generic UI
  operations remain with the UI provider and do not change resource ownership.
- Discovery MUST identify the owning provider and generation as well as panel
  identity. Updates and queued interactions MUST validate the target generation
  and runtime identity before affecting a live panel.

### Registration and lifetime

- Registration MUST stage under the declaring provider until commit. Acceptance
  is not activation; construction/attachment failures MUST have an explicit
  observable result, and MUST NOT appear as a successfully active panel.
- Attachment, visual mutation, and detachment MUST execute on the UI thread.
  Provider work MUST NOT block that thread waiting for work queued to itself.
- Retiring a provider MUST reject new panel work, invalidate its queued work,
  drain in-flight callbacks, and withdraw its panel resources. No retired
  generation may modify or remove a replacement generation's panels.
- All objects, callbacks, workers, and attachments that require provider code
  MUST be released before its DLL unloads. Deferred host-only visual cleanup
  may outlive the DLL only when it retains no provider code or data references.
- Unloading the UI provider MUST cancel its own requests and remove only its
  own panels. Other providers' panels and registrations MUST remain intact.
- Reload MUST NOT promise preservation of unsaved panel-local state without an
  explicit provider-owned persistence contract.

### Approved provider mapping

The migration preserves the seven existing root panels and their tab composition.
Schedule owns Calendar; Console Plugin owns Console; Settings Plugin owns Settings;
Model Workspace owns Training Control; Storage Plugin owns Shared Storage; DataHub
Plugin owns DataHub. GeoSpatial remains a migration target.

| Panel key | Owning provider |
|---|---|
| `calendar` | Schedule |
| `console` | Console Plugin (migrated) |
| `settings` | Settings Plugin (migrated) |
| `training_control` | Model Workspace Plugin (migrated) |
| `data_hub` | DataHub Plugin (migrated) |
| `storage` | Storage Plugin (migrated) |
| `geospatial` | GeoSpatial (new) |

Model Workspace retains Home, Knowledge Gaps, Tool Gaps, Model Config, Tokenizer,
Observatory, World Lab, Assets, and Tools in the existing Training Control panel.
World Lab does not become a separate provider or root panel in this migration.
DataHub also retains its existing tabs. Owning a composed panel does not transfer
the domain services it consumes to that provider.

### Implementation boundary

Providers own concrete `UIPanel` subclasses and their native widgets and callbacks.
They import the existing host UI implementation through its exported C++ interface.
Native UI plugins require the matching compiler, CRT and UI class ABI; API v14
rejects older binaries. Domain services remain with their respective providers.

Calendar in Schedule, Console in Console Plugin, Settings in Settings Plugin, Training Control in Model Workspace, Shared Storage in Storage Plugin, and DataHub in DataHub Plugin use this path. Constructor registration stages
ownership under the loader's initialization scope. Commit attaches the same object;
retirement detaches it before provider shutdown destroys it. No JSON widget tree,
interaction-event transport or generic copied-content panel remains.

### Current gap

The UI plugin still declares GeoSpatial through `register_ui_panel`.
Its provider migration remains outstanding. Dynamic surfaces and OSINT keep
their own lifecycle paths. The generic `ui.list` response still lacks owner
metadata; World Lab reads provider ownership from the resource registry.

## Current panel identity

`UIPanel` has two independent identities:

- `getPanelID()` is the existing numeric runtime identity for focus/input routing.
  It is generated for each instance and is not a persistent address.
- `getPanelKey()` is an immutable string supplied as the third constructor
  argument. It addresses the panel independently of its mutable display title.
  A recreated built-in panel reuses its semantic key and receives a new runtime ID.

The host UI must first be initialized on its owning UI thread, as in
`bootstrap/ui_bootstrap.cpp`. Panel keys are explicit application conventions;
they are not generated from titles and require no database.

## Built-in keys

| Key | Panel |
|---|---|
| `console` | Console |
| `settings` | Settings |
| `training_control` | GRIM-text Training Control (including World Lab and Assets tabs) |
| `data_hub` | DataHub |
| `storage` | Shared Storage |
| `calendar` | Calendar |
| `geospatial` | GeoSpatial |

These are the seven current provider-declared panels. Calendar belongs to
Schedule; Console belongs to Console Plugin; Settings belongs to Settings Plugin; Training Control belongs to Model Workspace; Shared Storage belongs to Storage Plugin; DataHub belongs to DataHub Plugin; GeoSpatial still belongs to the UI plugin. Physical and digital
environment panels are no longer registered; the Model Registry panel source is
excluded from the host build.

Discovery reports instantiated panels, not factories for panels that have not
been created. Tabs are not separate root panels. Existing unkeyed panels remain
usable by legacy host code but are excluded from stable-key discovery. OSINT
panel content and command behavior are outside this change.

`UIRoot::addPanel` registers a nonempty key and rejects/logs a duplicate without
replacing the original. Adding the same object twice is idempotent. Distinct
keys may have identical titles. Removing a panel releases its key for recreation.

## Host operations

The host-facing C++ interface is in `ui/ui_root.hpp`; panel management and request
execution live in `ui/ui_root_panels.cpp` so they can be tested without starting
the renderer or application.

- `getPanelByKey(key)` performs an exact lookup. Its returned live object is for
  host use; lookup locking does not make arbitrary panel mutation thread-safe.
- `requestPanelList()` returns a future of copied, key-sorted `UIPanelInfo`
  records: key, title, runtime ID, visibility, minimized/maximized state,
  position, size, and Z-order.
- `requestPanelOperation(key, operation)` supports `Inspect`, `Show`, `Hide`, and
  `BringToFront`. It returns a future of `UIPanelOperationResult`, containing
  success/error and a post-operation snapshot on success. Errors are `not_found`
  or `invalid_operation`. Operation requests accept keys only, never titles.

Requests made on the UI thread execute immediately. Other threads enqueue work;
their futures complete after the UI lifecycle processes that work. Do not block
the UI thread waiting for a queued future. A target removed before execution
returns `not_found`; requests address the current instance at a key, rather than
pinning the instance that existed when the request was submitted. Show sets
visibility and brings the panel forward; it does not restore minimized state.
Bring-to-front alone does not show a hidden panel. No operation synthesizes
pointer clicks or alters the numeric input-routing identity.

`processPendingTasks()` is the owning-thread lifecycle pump used by `update`;
calling it on another thread is rejected. Pending queued tasks are discarded
on shutdown (their futures report a broken promise). Requests require an active
UI lifecycle; there is no background UI executor or timeout mechanism here.

Legacy `getPanel`, `setVisible`, `removePanel`, and `bringToFront` accept a key
first or a unique current display title as a compatibility path. Ambiguous titles
do not resolve. Legacy void mutation methods remain fire-and-forget; new callers
needing completion should use the future-based interface.

## Dynamic surfaces

API v3.14 also supplies `GRIM_UI_HIDE_CURRENT` and optional `expected_runtime_id`
on `ui.hide`, applying the same UI-thread stale-instance check before hiding.
World Lab accepts `ui.hide <observed name or panel:reference>` and waits for its
completion. Native witness replay rejects hide just like other host actions.

For World Lab's resource directory, `submitPanelRequest` additionally accepts an
optional expected runtime identity. It checks the current instance on the UI
thread before applying the operation. A mismatch returns `stale_panel` with no
effect. API v3.13's `GRIM_UI_SHOW_CURRENT` carries that identity through the plugin
boundary. `ui.show` accepts `expected_runtime_id`; omission retains stable-key-only
behavior. `GRIM_UI_CANCEL`/`ui.poll` with `cancel:true` cancel still-queued owned work
and release the request; they cannot undo already completed effects. See
[manual World Lab dispatch](../WORLD_LAB.md#manual-ui-resource-discovery-and-dispatch).

`DynamicSurfacePanel` uses its existing `UISurfaceSpec::surface_id` as its panel
key. The renderer bridge addresses it by key, so different or renamed titles
do not break lookup. Create/update callbacks defer registry reads to the UI task
queue to avoid querying the registry while its notification mutex is held.
Initial rendered visibility follows the registered spec. Generic show/hide also
updates dynamic surface registry visibility. Registry lifecycle events only
mutate dynamic panels, so a surface key colliding with a host panel cannot hide
or remove that host panel; the bridge logs and rejects rendering that collision.

Surface creation/destruction remains owned by `UISurfaceRegistry`. The generic
panel operation interface deliberately exposes neither creation nor destruction.
Dynamic keys persist across restarts only if a caller saves/restores the surface
definition; this foundation adds no surface persistence.

## Plugin migration boundary

API v3.8 exposes the foundation through `ui_panel_service` and the folder-loaded
UI Plugin. The plugin registers these JSON tools:

```text
ui.list {}
ui.inspect {"panel_id":"settings"}
ui.show {"panel_id":"settings"}
ui.hide {"panel_id":"settings"}
ui.bring_to_front {"panel_id":"console"}
ui.poll {"request_id":1}
```

The first five operations always queue host work and return a pending response
with a request ID. Poll after frame processing; a completed response contains
`result.success` plus metadata or a failure. Polling a completed result consumes
it. The plugin's initial success confirms acceptance only. No handler waits for
the frame thread. Requests belong to a plugin instance, are capped at 64 per
instance, and are cancelled before execution if that instance unloads.

The old `console`, `toggle_settings`, `ui.show_surface`, and `ui.hide_surface`
registrations are removed; their handler logic remains for host compatibility.
Surface creation/destruction handlers remain host-owned but unregistered. No OSINT migration is
included. See [API completion/ownership semantics](../PLUGIN_API.md#generic-ui-operations-v38)
and [plugin build/deployment instructions](../../plugins/README.md#ui-plugin).

Focused validation with the installed toolchain:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1
```

`tests/ui_panel_operations_tests.cpp` exercises the real panel management,
request queue, surface registry, and renderer bridge with native window/render
effects stubbed. It covers identity, title changes/ambiguity, duplicate keys,
recreation, queued completion, removed targets, dynamic lifecycle and host-key
collisions. No GRIM executable, model server, training, or package installation
is needed.

Console now follows the native provider-owned path; its bootstrap factory and
UI Plugin factory declaration have been removed. See [Console ownership](CONSOLE.md#provider-ownership).

Settings also follows the native provider-owned path. See [Settings ownership](SETTINGS.md).

Storage follows the same native path and retains My Files and Devices. The plugin
owns widgets, drafts and presentation caches; the host retains DeviceCommServer,
its storage index, transfers and network lifecycle. UI-thread functions in
`ui/storage_panel_services.*` return copied snapshots and execute explicit device
operations without exposing a server pointer to the DLL. Network bootstrap binds
the service before plugin initialization; teardown unbinds after all plugins retire.
Missing services reject initialization with a logged error, not an empty panel.
Reload discards panel-local drafts but does not stop transfers or disconnect the hub.
Malformed explicit hub ports produce visible validation errors without calling
networking. The existing test command includes Storage DLL snapshot/action,
ownership, replacement, missing-service, permission and unload-before-activation tests.

DataHub follows the native path with all six existing tabs. See
[DataHub provider ownership](../../plugins/README.md#datahub-plugin) for host jobs,
reload-safe persistence and validation. The provider owns no worker threads.
