# World Lab and Assets: user and maintenance reference

Verified against the repository implementation on 2026-10-23.

This guide covers the environment-authoring system in **Training Control → World Lab**
and **Training Control → Assets**. Goal, Play, and Inspect are modes within
World Lab, not separate top-level tabs. The other Training Control tabs are
covered by [the panel README](../ui/UI_TRAINING_PANEL_README.md).

Physical navigation is the current pilot. The broader direction includes other
navigation/action tasks, such as simulated UI workflows and problem solving.
Those mechanics are not implemented yet; the current schema and worker still
use TextWorld's physical-world subset.

## Contents

1. [Quick workflow](#quick-workflow)
2. [World Lab controls](#world-lab-controls)
3. [Entity types and customization](#entity-types-and-customization)
4. [Goal conditions](#goal-conditions)
5. [Playtesting and solutions](#playtesting-and-solutions)
6. [Assets and presets](#assets-and-presets)
7. [Files, recovery, and limits](#files-recovery-and-limits)
8. [Troubleshooting](#troubleshooting)
9. [Terminology index](#terminology-index)
10. [Architecture and future changes](#architecture-and-future-changes)
11. [Validation workflow](#validation-workflow)

## Quick workflow

1. Open World Lab and choose New or a game from Saved games.
2. Create rooms with **+ Room**, or drag a compass handle into empty canvas space.
3. Connect rooms by dragging a compass handle onto another room.
4. Add objects using Inspect, or insert presets from Assets.
5. Select the starting room and choose **Set player start**.
6. Choose Goal, describe the objective, and add win/fail conditions.
7. Choose Play → Compile & Play, then enter commands to test the game.
8. Optionally Record a successful run and choose Save solution.
9. Save the world to retain the authored game and its verified solution.

The editor changes the authored starting state. Playing changes a separate
interpreter state. Picking up an object during play does not move it in the saved
world's starting layout.

## World Lab controls

World Lab and Assets have draggable pane dividers. Drag the vertical grip between
the canvas and inspector to adjust the inspector width. Drag the horizontal grip
above the Asset Library to adjust its height. Right-click a divider to restore
its default size; Escape cancels an active resize. Completed adjustments survive
tab switches for the current application session, but are not saved across
restarts or stored in world/asset files. Minimum sizes constrain the panes when
space permits; shrinking the outer panel does not erase the preferred sizes.

| Control | Behavior |
|---|---|
| World title | Changes the document title; does not rename its file. |
| New | Starts a one-room world and selects an unused default filename. Prior document changes remain available through Undo. |
| Open | Loads the path in the file field. Invalid files leave the current world intact. |
| Saved games | Opens a direct `.grimworld.json` child of the games folder. Labels derive from filenames. |
| Save | Applies valid edits and saves to the displayed path. Change this path to save another copy. |
| Undo / Redo | Restores document snapshots, including New/Open and whole asset insertions. File paths and filesystem operations are not document history. |
| + Room | Creates a room near the canvas center. |
| Fit | Frames all rooms. |
| Recover | Loads the per-user recovery snapshot and marks it unsaved. Save it to the intended world path. |
| Inspect | Shows properties for the selected room, object, or passage. |
| Goal | Shows objective and condition editing. |
| Play | Shows manual playtest controls and command history. |

**Canvas interactions**

- Click a room header or an object row to select it; click a connection line to
  select a passage.
- Drag a room header to move its card. Card position does not change compass directions.
- Drag a compass handle to a room to connect them; drop on blank canvas to create
  a new connected room. Exits are reciprocal: east from A means west from B.
- Drag an object row onto a room, container, or supporter to move it there.
- Middle-drag pans; the wheel zooms. Shift-wheel pans horizontally.
- With canvas keyboard focus: Ctrl+Z undoes, Ctrl+Shift+Z or Ctrl+Y redoes,
  Delete removes the selection, and Escape cancels a gesture.
- Text fields own their own keyboard input. Scroll the inspector to reach controls
  below the visible area.

**Inspector controls**

| Selection | Editable fields and actions |
|---|---|
| Room | Name, description, object type + Add here, Set player start, Delete selection, Save as asset. |
| Portable object/key/food | Name, description, Location, Delete selection, Save as asset. |
| Container | Name, description, Location, starting open/closed state, Add here, Delete selection, Save as asset. |
| Supporter | Name, description, Location, Add here, Delete selection, Save as asset. |
| Passage | Door name, starting door state, matching key, Delete selection. |

Apply changes or Enter commits name/description edits. Valid pending edits are
also applied by several navigation/save/recovery actions. A validation error
leaves the problematic field available for correction.

Deleting a holder also removes its contents. Deleting a room removes incident
passages. References in challenge conditions are cleaned up. Deleting a key clears
matching door references and changes a locked door to closed. Undo restores the
whole transaction. If the player-start room is deleted, another remaining room
becomes the start.

## Entity types and customization

| Type | Meaning | Useful customization |
|---|---|---|
| Room | A location the player can occupy. | Name, description, contents, compass connections, start position. |
| Object | A portable generic item. | Name, description, starting location. |
| Key | A portable item that may match a passage's door. | Name, description, location; choose it in the door's matching-key field. |
| Container | A holder that can be opened or closed. | Name, description, state, contents. |
| Supporter | A surface on which items rest. | Name, description, contents; examples include tables and benches. |
| Food | A portable item the player can eat. | Name, description, location; can be used with an eaten condition. |

A door is represented by properties on a passage, not by an ordinary entity.
An empty door name means an unobstructed passage. Named doors can start open,
closed, or locked; locked doors require a matching key.

Container locking is not exposed as an authoring mechanic: containers start open
or closed and have no matching-key field. Although the condition selector accepts
`locked` for containers, the current authoring subset cannot construct a useful
locked-container puzzle. Use a door for key-and-lock puzzles.

Descriptions provide narrative, not executable behavior. Naming an object
"lantern" does not add lighting, and describing seeds does not add a planting
command. Custom actions, characters, dialogue, scripts, physics, crafting, and
arbitrary properties are not exposed. Changing the type selector controls new
items; it does not convert the selected existing item to another type.

The editor allows more nesting than TextWorld can compile. For playable worlds,
place containers and supporters directly in rooms; put portable items in rooms,
inside containers, or on supporters.

## Goal conditions

The objective is player-facing text. The actual win/loss rules come from the
condition lists; writing a goal in the objective does not create those rules.

Choose a condition, subject, and optional target; then press Add win or Add fail.
Select a saved condition and Remove to delete it. Apply goal commits the objective.
To change a condition, remove the old condition and add its replacement.

| UI label | Stored predicate | Subject | Target | Meaning |
|---|---|---|---|---|
| Player is in room | `player_at` | Room | None | Player occupies that room. |
| Player carries | `carrying` | Object, key, or food | None | Item is in player inventory. |
| Object is in room | `at` | Object, key, or food | Room | Item is directly in the room. |
| Object is inside | `in` | Object, key, or food | Container | Item is inside the container. |
| Object is on | `on` | Object, key, or food | Supporter | Item rests on the supporter. |
| Is open | `open` | Container or named door | None | Selected holder/door is open. |
| Is closed | `closed` | Container or named door | None | Selected holder/door is closed. |
| Is locked | `locked` | Container or named door | None | Selected holder/door has the locked fact; see container limitation above. |
| Has been eaten | `eaten` | Food | None | Food has been consumed. |

**Every win condition must hold together. Any single fail condition can end the
run in failure.** Conditions are state tests, not ordered checkpoints. There is no
UI for OR groups of winning conditions, negation, timed objectives, or sequences.

Containment conditions are precise: an item inside a box is not directly on the
room floor. Use `in(item, box)` rather than `at(item, room)` for that goal.

Example: The Garden Delivery requires `player_at(Garden)` and
`carrying(seed packet)`. It has no fail conditions. A rule such as
`eaten(red apple)` could be added as a failure condition in another game.

No win conditions means no authored victory goal. A world with neither wins nor
fails is an exploration sandbox; a world with only fail conditions can still lose.

## Playtesting and solutions

**Inspecting and preparing actions**

Select an entity or passage, then choose **Actions** beside Apply changes to
scroll to its supported TextWorld rules. The same inspector is available in
Assets through **Actions** beside Apply. The catalog shows command templates,
typed parameter choices, required facts, removed facts, and added facts. These
are definitions from TextWorld 1.7, not an evaluation of the current play state.
Different rule IDs may share a command, such as the examination rules for items
in different locations. Container lock/unlock rules are visible with a notice
that matching-container-key authoring is not yet supported.

In World Lab, bind any remaining parameters and choose **Prepare command** to
open Play with the command in its editable field. Preparation does not execute
the action or change the authored starting state. Review it, then Send. Asset
inspection does not run commands; insert the asset into a world to playtest it.

In Play, **Actions** toggles a compact composer for all supported built-in rules.
Choose an action, bind its parameters, then Prepare command and Send. Entity
bindings use IDs; names are resolved when preparing the command. Free-text input
remains available. The **Live available commands** dropdown fills the command
field from the latest interpreter state, even while inspecting an older history
entry. It is hidden while busy, after completion, or when compilation is stale.

The built-in catalog is read-only. It is an offline snapshot generated from the
installed knowledge base; compilation checks it against the runtime catalog. A
mismatch blocks preparation in the Play composer and reports the discrepancy.
Custom action authoring and non-compass transitions remain future work.

| Control | Behavior |
|---|---|
| Validate (Goal) | Compiles and resets a temporary candidate interpreter, then closes it. Checks compatibility, not solvability; does not replace an existing live play session. |
| Compile & Play | Compiles the current authored world and starts its live interpreter from the beginning. |
| Command field / Send / Enter | Sends one text command to the live game. |
| Reset | Restarts the compiled game from its original state. |
| Stop | Stops the worker and cancels the session/request. Compile again to resume play. |
| Record | Resets first, then records subsequent commands. |
| Save solution | Replays recorded commands in a separate interpreter; stores them in the document only if replay reaches a win. Save the world file afterward. |
| Check saved | Replays the document's saved solution without changing the live interpreter. |
| History | Displays an earlier observation; does not rewind gameplay. |

Typical commands include `look`, `inventory`, `go east`, `open seed box`,
`take seed packet from seed box`, and `unlock garden gate with brass key`.
The available-commands list is an authoring aid for the current observation.

The canvas highlights the current room and changed objects. Its cards and object
lists still represent the authored world. History selection can show highlights
from an older observation; the next command always acts on the live state.

Changing semantic game data requires recompilation. Moving cards or changing
solution metadata does not. Leaving/hiding/minimizing World Lab stops its worker;
visiting Assets therefore ends the play session too. Save the world before
closing the application. A solution is a command list, not a transcript or model
training episode, and successful verification is evidence for that exact game.

## Assets and presets

World Lab has an **Asset Library** at the bottom of the panel. Saved assets
appear as scrollable tiles showing their type and entity count. Filter by the
asset's root type (room, object, key, container, supporter, or food), or search
names, descriptions, and nested contents. Results rank name matches above
contents matches; this is text relevance, not semantic similarity. The library
refreshes periodically and when returning to World Lab.

Drag a tile onto the canvas to insert an independent copy. Rooms are positioned
at the drop point without changing the canvas view. Other assets must be dropped
on a room header, container row, or supporter row. The drag label indicates
whether the target is valid. Escape, hiding the panel, or releasing outside a
valid target cancels placement. Each successful insertion is one undoable change;
save the world to persist it. Invalid asset files are labelled and cannot be
dragged. Use the Assets tab to create, edit, and save templates.

The **Assets** tab also includes this tiled library, with the same search and
type filters. Click a tile there to load its saved template into the editor.
Save or Discard edits before switching templates; a blocked selection preserves
the draft. Saving updates the opened asset file. Tiles in Assets do not insert
copies into the preview or world; use **Add to world** for world insertion.

Assets are reusable templates stored independently of games. A room asset includes
its contained objects. An object asset can include descendants, such as a chest
containing a key. Capture excludes passages, doors, goals, player-start state,
solutions, and original room coordinates.

| Control | Behavior |
|---|---|
| Saved assets | Browses room and object categories; loads a selected template. Invalid files are labelled. |
| Library name field | Names the browser entry. Saving this change does not rename its existing file or root entity. |
| Type selector + New asset | Creates a room, object, key, container, supporter, or food template. |
| Canvas / Inspector | Select contents; edit in-game names, descriptions, and container state. Pan/zoom are available. |
| Apply | Commits inspector edits to the asset draft. |
| Type selector + Add inside | Adds a new object to the selected room, container, or supporter. |
| Remove item | Deletes a selected descendant and its contents; cannot remove the asset root. |
| Save | Persists the current template; first save chooses a generated filename. |
| Save copy | Persists a separate template under an unused filename. |
| Add to world | Inserts the current draft as an independent copy into World Lab and switches there. Does not save either file automatically. |
| Undo / Redo | Traverses asset-edit history. Opening another asset starts a separate history. |
| Discard edits | Restores the saved/opened/created baseline and clears history. |
| Delete asset | Moves the saved file to `.trash`; does not remove instances from games. |

For object insertion, first select a destination room, container, or supporter in
World Lab. For room insertion, the new card appears near the canvas center. Each
insertion allocates destination IDs, remaps parents, and suffixes conflicting
names. It is one undoable operation. Updating a library preset never updates
previously inserted copies.

The Assets canvas is a selection preview: it does not expose room connections,
object reparenting gestures, or world keyboard-edit shortcuts. Standalone objects
appear in a synthetic preview room which is excluded from the saved asset.

Dirty drafts block replacement actions until saved or discarded. Switching tabs
preserves fields in memory, but assets have no recovery autosave across app exits.
A newly captured, unedited template is labelled New asset and still needs Save;
it is not necessarily considered dirty just because it has no file yet.

**Starter presets**

| Category | Preset | Included contents |
|---|---|---|
| Room | Cozy Cottage | Kitchen table, bread loaf, closed pantry cupboard, clay cup. |
| Room | Small Workshop | Workbench, mallet, closed tool chest, iron key. |
| Room | Walled Garden | Garden bench, watering can, sunflower seeds. |
| Object | Supply Chest | Closed chest, brass key, folded map. |
| Object | Writing Desk | Desk, notebook, pencil. |
| Object | Red Apple | Edible portable food. |
| Object | Brass Lantern | Portable generic item; no lighting mechanic. |

Preset keys need to be assigned to a door in the destination game before they
unlock anything. Templates do not carry challenge rules.

## Files, recovery, and limits

Paths below are relative to the repository's resolved resource directory unless
stated otherwise. In this checkout that directory is `D:\G.R.I.M\resources`.

| Data | Location / format |
|---|---|
| Games | `models/games/*.grimworld.json`; format `grim-world`, version 2. Version 1 can still load. |
| Room templates | `models/world_assets/rooms/*.grimasset.json`; format `grim-world-asset`, version 1. |
| Object templates | `models/world_assets/objects/*.grimasset.json`; same asset format. |
| Deleted templates | `models/world_assets/.trash/<original filename>.<timestamp>`. |
| World recovery | `%LOCALAPPDATA%/GRIM/WorldLab/recovery.grimworld.json` on Windows. |
| Compiled games / worker logs | Owned temporary session directory; removed at worker cleanup. |

To restore a deleted template, move it back to its original category and remove
the appended timestamp so its filename ends in `.grimasset.json` again. There is
no restore button. Avoid overwriting a newer template with the same filename.

Libraries refresh about every two seconds while active and their dropdown is
closed, and after relevant saves. Games scan direct children only; assets scan
only the rooms/objects category directories. They do not recursively search arbitrary
folders. Custom world file paths remain supported if the parent directory exists.

World recovery checkpoints valid changes every ten seconds and when hiding the
view. It is one per-user snapshot, not a backup history or a substitute for Save.
An untouched new session does not overwrite an older recovery snapshot. World
Undo restores document content, not the previously selected save path: check the
path after undoing Open/New before saving.

| Boundary | Current limit |
|---|---|
| JSON input file / transport payload | 16 MB. |
| World document | 10,000 entities; 20,000 passages. |
| Asset tree | 1–9,999 entities, with one root. |
| TextWorld compilation | 1,000 entities; 2,000 passages. |
| Conditions | 64 win and 64 fail conditions. |
| Objective | 4,096 bytes in native document validation. |
| Name / description in document | Name 1–256 bytes; description up to 16,384 bytes. |
| Playable names | Unique case-insensitively; 1–80 ASCII characters, starting with a letter; letters, digits, spaces and hyphens only. |
| Episode / saved solution | Up to 1,000 commands; solution verification needs at least one. |
| Commands | Native stored commands limited to 512 bytes; worker checks 512 characters and rejects line breaks/NUL. |
| Undo snapshots | Up to 100 committed edits. |
| Worker request | One in flight; 120-second timeout; worker idle exit after 15 minutes. |

TextWorld text validation rejects quotes, square brackets, backslashes, line
breaks, and NUL in strings embedded into Inform source. A valid editor document
is not necessarily a playable TextWorld game. JSON saves use a sibling staging
directory and atomic destination replacement; failed writes preserve the old file.

## Troubleshooting

| Symptom | Check / action |
|---|---|
| Game or asset missing from dropdown | Check category, extension, and folder; close the dropdown and wait for refresh. |
| Add to world fails for an object | Select a valid holder in World Lab first; apply valid pending world edits. |
| New/Open/capture blocked in Assets | Save or Discard the edited asset draft. |
| Lower inspector controls missing | Scroll the inspector or enlarge the panel. |
| Game saves but fails compilation | Check playable names, duplicate names, forbidden text characters, nesting, spawn, and door-key references. |
| Locked door cannot be opened | Ensure its matching key exists, is reachable, and can be carried. Unlock it before opening. |
| Goal never wins | Check all win conditions together, direct vs nested location, and contradictory conditions. Objective prose is not a rule. |
| World changed / recompile message | Compile & Play again after semantic edits. |
| Commands unavailable after tab switch | The worker stopped on leaving World Lab; compile again. |
| Save solution rejected | Record from reset, reach a win, and omit commands after the episode ends. |
| Docker startup fails | Check the existing Docker Desktop Linux engine and `grim-textworld:1.7.0` image; consult TEXTWORLD_SETUP.md. |

## Terminology index

| Term | Meaning in this system |
|---|---|
| Admissible commands | Actions reported by TextWorld as available for an observation; shown as an author aid. |
| Asset | A reusable independent entity-tree template. |
| Asset root | The one room/object from which all saved asset descendants originate. |
| Authored state | Starting entities, locations, door states, and rules saved in the world. |
| Canvas | The pannable/zoomable room-card workspace or asset preview. |
| Goal | Objective text plus win/fail state conditions. |
| Compile | Translate the authored game into a temporary playable TextWorld artifact. |
| Condition | A predicate with a subject and, where needed, target. |
| Container | An openable/closable holder with contents inside. |
| Descendant | An entity inside another entity, directly or through further holders. |
| Dirty | Draft or field content differs from the editor's baseline. |
| Door | Named stateful properties attached to a passage. |
| Entity | A room, object, key, container, supporter, or food with a stable ID. |
| Episode | One run from reset until completion or reset; not currently a training record. |
| Fact | A TextWorld state proposition such as an item's location or a door being open. |
| Holder | A room, container, or supporter that can parent an object. |
| Inspector | Controls for the selected entity or passage. |
| Instance / copy | Independent entities inserted into a world from an asset. |
| Inventory | Portable items currently carried by the player. |
| Layout | Canvas coordinates, separate from logical world connectivity. |
| Live state | Current interpreter observation after player actions. |
| Parent | The ID of an entity's containing holder. |
| Passage | A reciprocal compass connection between two rooms, optionally with a door. |
| Portable | Generic object, key, or food; excludes rooms, containers, and supporters. |
| Predicate | The condition operator, such as carrying, in, or player_at. |
| Preset | A preauthored asset provided for reuse. |
| Recovery | The single per-user world checkpoint opened with Recover. |
| Semantic edit | Change to game data included in gameJson; invalidates compiled-session compatibility. |
| Snapshot | A complete document state used by transactional undo/redo. |
| Solution | A saved list of commands that can be replayed to verify victory. |
| Spawn | The room ID chosen as the player's initial location. |
| Subject / target | Entity being tested / optional destination of a condition. |
| Supporter | A surface that holds items on it. |
| Transaction | Validated document mutation committed as one history step. |
| Validate | Compilation/reset check; does not prove a winning path exists. |
| Worker | The isolated Python/TextWorld process controlled through Docker and JSON mailboxes. |

## Architecture and future changes

### Code ownership map

| Source | Responsibility / when to change it |
|---|---|
| `ui/ui_training_panel.hpp`, `.cpp` | Top-level tab registration, view ownership, dispatch, and cross-tab callbacks/lifecycle. |
| `ui/world_lab/world_document.hpp`, `.cpp` | Entity/passage/condition schema, structural validation, ID allocation, history, version migration, JSON persistence. |
| `ui/world_lab/ui_world_lab_view.hpp`, `.cpp` | World toolbar, inspector, game discovery, recovery, selection capture and asset insertion. |
| `ui/world_lab/ui_world_canvas.hpp`, `.cpp` | Cards, entity rows, selection, drag gestures, compass connections, highlights, selection-only asset mode. |
| `ui/world_lab/world_asset.hpp`, `.cpp` | Asset schema, extraction, validation, synthetic previews, insertion/remapping, naming, file discovery. |
| `ui/world_lab/ui_world_assets_view.hpp`, `.cpp` | Independent asset editor, library controls, draft protection, save/copy/archive. |
| `ui/world_lab/ui_world_session.hpp`, `.cpp` | Goal UI, play controls, async responses, command history and solution recording. |
| `ui/world_lab/world_actions.*`, `ui_world_actions.*` | Shared catalog access, typed ID binding, action inspection and command composition. |
| `ui/world_lab/textworld_action_catalog.generated.hpp` | Generated offline TextWorld rules; do not edit by hand. |
| `scripts/textworld/action_catalog.py` | Exports native TextWorld rules for offline inspection and runtime compatibility checks. |
| `ui/world_lab/textworld_client.hpp`, `.cpp` | Hidden Docker process, asynchronous request lifecycle, timeouts, cancellation, mailbox transport and cleanup. |
| `scripts/textworld/world_worker.py` | Native-to-TextWorld translation, compiler/interpreter operations, facts, win/loss conditions and solution replay. |
| `ui/primitives/ui_canvas.*`, `ui/input/ui_input_router.hpp` | Shared canvas transforms, gestures, focus, capture and popup routing. |
| `resources.cpp` | Resolves resource root; UI passes library directories to views. |
| `cmake/Sources.cmake` | Discovers UI sources with CONFIGURE_DEPENDS; new host-test sources still need explicit registration. |

### Data flow

World controls → Editor transaction → Document → gameJson → TextWorldClient →
Python worker → compiler/interpreter → response → session UI and canvas highlights.

World selection → Asset.capture → independent asset draft → `.grimasset.json` →
Asset.insert → one destination Editor transaction. The original game, library file,
and inserted instance do not share mutable state.

World JSON stores `format`, `version`, `title`, `spawn`, `next_id`, `entities`,
`passages`, `editor.rooms`, `objective`, `win_conditions`, `fail_conditions`, and
`solution`. Entity entries contain `id`, `name`, `kind`, `parent`, `description`,
and `state`. Passages contain endpoint IDs, direction, and door state/key data.
Conditions contain `predicate`, `subject`, and `target`.

Asset JSON stores `format`, `version`, `name`, `root`, and `entities`. IDs in an
asset are local to that asset and must be remapped on insertion. The synthetic
preview room is editor scaffolding, never part of an object template on disk.

`gameJson()` excludes editor coordinates, next-ID bookkeeping, and solution
metadata. The session compares this representation against its compiled snapshot.
Keep that boundary accurate when adding fields: gameplay changes must invalidate
old sessions, while purely visual edits should not.

### Invariants to preserve

- Use `Editor::change` for transactional edits; failed validation must leave the
  current document intact. Keep multi-entity operations in one undo step.
- Maintain unique IDs, valid parents, acyclic containment, one layout per room,
  valid spawn, reciprocal unoccupied exits, and valid condition/key references.
- Keep deletion cleanup synchronized with every new ID-reference field.
- Preserve independent asset copies, destination ID allocation, parent remapping,
  duplicate-name handling, and failure rollback. Do not copy external references.
- Keep asset editing history bounded to the currently opened asset; use reset at
  document boundaries. Keep root identity valid after Undo/Redo and deletion.
- Preserve pending input on validation failures. Do not refresh fields from stale
  snapshots or silently discard edited drafts.
- Route expanded dropdown input first, respect panel/inspector clips, and release
  popup/focus/capture ownership when hiding views. Asset previews must not allow
  world topology edits or deletion of their synthetic room/root.
- Keep file replacement atomic and bounded reads intact. Respect Windows paths
  and C++20 UTF-8: `path.u8string()` returns `std::u8string`, not `std::string`.
- Keep Docker waits off the UI thread, one request in flight, bounded payloads,
  request IDs/version checks, cancellation, and owned temporary-file cleanup.
- Do not import model/training dependencies into the world worker. Its network is
  disabled; source is mounted read-only, with only the session directory writable.

### Change checklist by feature

| Proposed change | Required coordination |
|---|---|
| New entity kind or behavior | Document validation and JSON, world/asset type selectors, rendering, asset insertion, worker type/action support, tests. A description alone cannot implement behavior. |
| New condition | Native condition validation, predicate labels and subject/target filtering, worker fact translation, deletion cleanup, tests and this reference. |
| New stored field | Defaults, parse/serialize compatibility, version decision, gameJson semantics, asset capture policy if applicable, round-trip tests. |
| New door/key rule | Passage schema, inspector, worker fact wiring, reference cleanup, solvability tests. |
| New asset capability | Asset format/version, extraction scope, preview representation, validation, remapping, independent-copy guarantees. |
| New tab/control | Registration, update/draw layout, focus/capture lifecycle, hide/minimize handling, host input tests. |
| New worker operation | Client request lifecycle, worker protocol, structured error handling, timeout/cancel behavior, stale-world responses, integration tests. |

Version 1 worlds load into the version 2 model; current saves emit version 2.
Asset format is separately versioned at 1. Do not silently reinterpret older files
when changing semantics. Preserve user-authored games/assets and unrelated changes.

Future model-driven gameplay, training episode export, curriculum generation,
linked asset updates, multi-room templates, asset tags/search, and arbitrary
TextWorld game import are not implemented. Treat them as new scoped work, not as
existing features to build assumptions around.

### Recurrent model interaction contract (selected; integration pending)

The [GRIM reasoning contract](GRIM_REASONING_DATA_CONTRACT.md) governs the model
side of future challenge assignment. The Goal describes the desired outcome,
success criteria, and constraints. The TextWorld quest supplies executable
environment checks. Align them explicitly; criterion prose does not become a
predicate automatically, and not every GRIM constraint is an environment fact.

Each model cycle follows Determine, optional Define, optional Execute, and
Answer, with optional Update where appropriate. Define and Execute need not
occur together. They are working context observed at span boundaries, not
TextWorld commands. Only the contents of the newly completed Answer span are
validated as one command and submitted to the environment. Submission must be
tracked once per answer instance; cumulative decoded trajectories must not be
resent as commands.

The returned observation and verified status become permitted context before the
next cycle. A requested transition is not an observed result. Retain the active
Goal while the challenge continues; emitting Answer does not prove victory or
end the challenge. Observations must not leak hidden world facts, saved solutions,
or future results. Ordinary user-facing responses use the same optional-phase
contract without necessarily submitting anything to an environment or tool.

Span-boundary pause/resume exists in GRIM inference. Answer dispatch, runtime
feedback insertion, challenge sequencing, and recurrent episode export remain
pending. The current source ConceptBlock stores one instance of each phase;
neither a whole challenge nor every environment step is assumed to map 1:1 to
an SFT block. This describes the intended integration, not new Play-tab behavior.

## Validation workflow

Read the repository AGENTS.md and applicable local instructions before changes.
The current build policy permits focused host utilities/tests but prohibits
configuring, building, linking, or running `grim.exe`, training/runtime targets,
model servers, or training loops without the user's specific authorization.
Do not install packages or mutate dependencies unless explicitly requested.

From the repository root, the existing host-only regression command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -WorldLab
```

This uses renderer sinks, not the full application. Relevant suites:

- `tests/world_lab_tests.cpp`: document validation, persistence, gestures,
  inspector interactions, game library, recovery, goal editing.
- `tests/world_asset_tests.cpp`: extraction, invalid trees, persistence, repeated
  insertion, fresh IDs/names, atomic undo, dirty protection, library and archive UI.
- `tests/ui_input_foundation_tests.cpp`, `tests/ui_canvas_tests.cpp`: shared input,
  clipping, popup ownership and canvas behavior.

For changes to TextWorld behavior or transport, use the existing installed Docker
image and the focused integration workflows in [WORLD_LAB.md](WORLD_LAB.md#validation).
`tests/test_world_worker.py` exercises compiler/interpreter semantics;
`tests/world_lab_client_tests.cpp` exercises native transport. Do not pull/build
images or start unrelated services as a side effect.

For documentation-only changes, verify paths, controls, schema names, and links;
a runtime build is unnecessary. For UI changes, host tests and syntax checks are
not visual proof: report that limitation and leave full app build/run to an
explicitly authorized step.

If future work creates or transforms ConceptBlock reasoning examples or changes
concept supervision, first read `docs/GRIM_REASONING_DATA_CONTRACT.md` and update
it when the documented contract changes. The current game and asset formats are
not themselves ConceptBlock reasoning data.

Related: [World Lab implementation notes](WORLD_LAB.md),
[TextWorld setup](TEXTWORLD_SETUP.md),
[asset library README](../resources/models/world_assets/README.md).
