# World assets

The Training Control Assets tab stores independent room and object templates as
`*.grimasset.json` in `rooms/` and `objects/`. Rooms include their contents; objects
include nested contents. Inserting an asset makes an independent copy with fresh
IDs and unique in-game names. Original passages and goals are not copied.

Use Save as asset in World Lab to capture a selection, then Save in Assets.
Use Add to world to insert a room, or select a destination room/container/supporter
in World Lab before inserting an object. Save the destination game afterward.

Delete asset moves the file to `.trash/`; restore it to its original subfolder to
recover it. Editing the library name renames the browser entry; Save copy creates
a separate file. Existing games never change when a library asset is edited.

## GRIM subsystem rooms

The room library includes **GRIM - System Hub**, **GRIM - Perception**,
**GRIM - Tools**, **GRIM - Computer**, and **GRIM - Memory**. Each has a room
description and fixed, inspectable reference stations grounded in the subsystem
interfaces. Stations use the existing `supporter` kind so they cannot be taken
as inventory. Use the current TextWorld `examine <name>` command to inspect them.
Their content is authored reference material; it does not expose live readings,
execute tools, control the desktop, or read/write actual memory.

Insert the rooms into World Lab, then select the System Hub and enable path
navigation to make it `/`. Assign the subsystem rooms `/perception`, `/tools`,
`/computer`, and `/memory` through their Room path fields. Assets do not persist
world-specific addresses, connections, Goals, or player state. Set the player
start and save the assembled world separately.

## Optional binding metadata

Room and object entities share optional metadata in both `grim-world-asset` v1
and world documents. Existing files remain valid; absent fields are omitted on
save. These fields describe bindings only: they do not load plugins, register
tools, dispatch actions, or create live/sandbox runtime state.

- Room `plugin_binding`: `provider_name` and `contract` (nonempty strings), plus
  `version` (positive uint32 contract version). No loaded provider is required
  to edit or validate a template.
- Non-room `resource_binding`: `type` and `key` (nonempty strings). The key is
  independent of generated world IDs and display names; duplicate room instances
  can preserve the same resource key. Environment/room scoping is for a future
  resolver to enforce.
- Any entity `asset_source`: `asset_id`, `entity_id` (nonempty source identifiers),
  and `version` (positive uint32 source asset version). This is explicit provenance,
  not a reference to the destination world entity. It is not automatically populated.
- Any entity `initial_state`: a JSON object of authored state metadata. Its contents
  have no engine semantics yet; container `state` retains its existing meaning.

Binding/provenance strings are limited to 256 bytes and exclude CR, LF, and NUL.
Capture, preview, insertion, editor undo/redo, and save/load preserve the metadata.
Insertion remaps world IDs and names while retaining binding keys and source IDs;
initial-state metadata is copied independently. Typed resource kinds, state
transitions, inventory registry projection,
and production/sandbox resolution are separate future work.

## Bound plugin rooms and resources

`rooms/plugin_*.grimasset.json` supplies one room for every provider configured in
`plugins/CMakeLists.txt`. `plugin_binding.provider_name` exactly matches the native
plugin info name used by tool registration. The shared `grim-plugin-room` v1
contract identifies this authored room/resource layout; it is not an executable
plugin API or sandbox action implementation.

Each room includes resource objects, also available independently as
`objects/plugin_*.grimasset.json`. Insert a room with its resource fixtures, or
select a destination plugin room and insert a standalone resource. Containment
(`parent`, walking ancestors for nested objects) supplies the room/provider
association; object assets do not duplicate provider names or tool registrations.
Resources use the existing fixed `supporter` kind so they remain in their rooms;
`resource_binding.type` describes their domain identity without adding native
TextWorld entity types. Asset provenance uses stable authored identifiers, not
library filenames or destination IDs.

The UI room includes `ui.panel:console` (visible) and `ui.panel:settings` (hidden),
plus a panel-request resource. These are authored initial fixtures, not captured
production visibility. Other resources have empty initial-state objects: no live
paths, hardware readings, aliases, memory contents, credentials, microphone data,
or network targets are inferred. Calculator and Utility have no resource objects:
their current handlers do not retain resource state. In particular, Utility clean
returns a fixed acknowledgment; it does not mutate console visibility/history.

The registry remains the source for tools, usage classifications and permissions.
These assets do not make Debug/System/Ignore tools model-selectable, mount the
registry in inventory, load plugin DLLs, or activate sandbox dispatch. Existing
`preset_grim_*` reference rooms and saved worlds remain independent and unchanged.
Provider compatibility and action/state semantics still require the resolver.
