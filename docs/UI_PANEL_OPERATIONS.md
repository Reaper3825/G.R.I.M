# Stable panel identity and host operation foundation

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
| `geospatial` | GeoSpatial |
| `physical_environment` | Physical Environment |
| `digital_environment` | Digital Environment |
| `model_registry` | Model Registry, when instantiated |

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
Surface creation/destruction remain host-owned tools. No OSINT migration is
included. See [API completion/ownership semantics](PLUGIN_API.md#generic-ui-operations-v38)
and [plugin build/deployment instructions](../plugins/README.md#ui-plugin).

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
