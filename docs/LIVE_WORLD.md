# Live System World

World Lab's **Live** button opens a persistent host-owned session. System Hub at `/` is permanent; every registered provider automatically gets a
child room on startup and as providers load. With no providers, only the hub remains. It does not compile or launch TextWorld, a model, or
a trainer. **World editor** returns to the authored world; entering Live again
resumes the same live session. Hiding World Lab preserves that session too.
Sessions currently last for the lifetime of the World Lab view; room identities and
history are not persisted across application restarts.

## Provider ownership

ToolRegistry publishes an explicit provider record and its tools under one mutex
and revision. Plugin initialization stages metadata; commit publishes both,
including providers with zero tools. Failed initialization or replacement leaves
the old publication unchanged. Shutdown removes only the matching generation.
The plugin ABI is unchanged. Built-in provider availability is derived from their
tool registrations. Direct legacy plugin registrations without a lifecycle commit
do not advertise an independently loaded provider.

The atomic snapshot contains copied descriptors, not handlers or DLL references.
Provider identity is `(provider_type, provider_name)`; `provider_instance` guards
the loaded generation. Late registrations must match the published generation and
source path. Removing an individual tool does not unload its provider.

## Connections and navigation

World membership follows registered availability automatically. There is no manual
Connect or Disconnect step. **Available actions** lists navigation and read-only
tool inspection commands.
Provider rooms are arranged radially around System Hub with spoke connections.
The compact ring uses one quarter of the original radius and expands with provider count; topology
changes redistribute the ring without changing room identities or player location.
Only adjacent navigation is allowed: return to the hub before entering a sibling.
Internal room IDs remain separate from navigation. Paths use provider names, e.g.
`/builtin` and `/UI`. A trailing "Plugin" label (space, hyphen, or underscore
separated, case-insensitive) is omitted from room names and paths. `go UI` uses
the shortened name; `go UI Plugin` still accepts the registered name. Internal
provider ownership and tool descriptors keep their registered identity.
Path-unsafe characters become hyphens. Collisions receive an identity-derived
suffix; paths stay reserved across unload/reload and never silently change owners.
Ambiguous exact names require the listed path. Room IDs and paths remain stable
through generation changes. Asset-library files do not
create availability or implicitly add resource objects.

Registry changes are reconciled on the UI thread before actions and while the
view is visible. A replacement preserves the room and player location, and updates
its tool catalog. Unload removes the connection. If that room is occupied, the
player returns to System Hub with an explicit history event. A subsequent load automatically restores the provider room with its existing
session room identity. Reconciliation on reopening handles changes made while the view was
hidden. These operations preserve session identity, movement count and history.

## Actions and scope

Supported commands: `look`, `tools`, `go /builtin`, `back`, and
`inspect <canonical_tool_id>`. Inspection uses the current provider generation and
only Loaded, Model-classified tools. Invalid or stale selections produce rejected
action entries. Tool contracts are copied from ToolDescriptor, including argument
schema, permissions, confirmation metadata, and bindings.

This foundation never executes provider tools. It does not project live resource
objects, provide sandbox transitions, or generate verified TextWorld witnesses.
Those require explicit resource/action adapters and policy checks. The existing
compiled TextWorld backend remains the authored-world playtest path. The live
canvas uses the existing document renderer in selection-only mode; its topology
is owned by the live session, not by editor gestures or undo history.

## Validation

`tests/run_live_world.ps1` builds and runs a focused host-only test executable.
It covers hub-only empty-registry startup, existing and newly loaded provider population,
zero-tool providers, adjacency,
Model filtering, inspection without handler invocation, replacement rollback,
stable room identity, stale revision rejection, unload/relocation, automatic reconnection, persistent history, and concurrent atomic snapshot reads.
Plugin lifecycle fixtures additionally cover provider commit/reload/unload.
`tests/run_ui_input_foundation.ps1 -WorldLab` exercises automatic room population and navigation
controls, reconciliation after hiding, and repeated client stop, alongside the
existing editor and asset tests.
No GRIM application, model, trainer, or TextWorld container is required.
