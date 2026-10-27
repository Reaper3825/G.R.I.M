# World Lab — reusable environments and Goal-driven manual episodes

For the full control reference, customization guide, terminology index, and agent
maintenance notes, see [World Lab and Assets reference](WORLD_LAB_REFERENCE.md).

World Lab is an authoring tab in the Training Control panel. It stores an editable
world locally. Validate world and Compile / Start start an isolated TextWorld Docker
worker on demand. No model, inference service, or training process is started.

Courses store an optional environment reference as `Course::environment_id`,
persisted in each course entry in `curriculum_registry.json`. An empty string
means no environment is assigned; older registries without the field load with
that default. Saved worlds now persist an `environment_id`; the environment
registry resolves it to a world file and checks that the file contains that ID.
Document entity IDs identify entities within a world; worker `environment-N`
filenames are temporary session artifacts and must not be used as course IDs.

### Assign and resolve an environment

1. Save a world in World Lab. Save registers its persistent ID and file location
   in `models/games/environment_registry.json` under the configured resources root.
2. In DataHub's Curriculum tab, select a course. Use **Courses → Refresh
   Environments** to discover saved worlds, then choose its **Environment**.
   Selection persists the ID immediately through the existing course registry.
3. Choose **No environment** to clear the assignment. Multiple courses can select
   the same environment.

`EnvironmentRegistry::resolve(id)` returns a validated location, and `load(id)`
loads the matching document. `UIWorldLabView::openEnvironment(id)` opens it in
the editor. Course references use the same ID with these APIs; automatic
course-to-episode assignment remains pending.

New worlds receive independent IDs. Save, title changes, and file moves retain
identity. **Save copy** creates an independent ID and an unused `-copy` filename;
changing the path alone cannot register a second live file with the same ID.
Editing world content preserves its ID but changes compilation/witness content
compatibility. The ID is excluded from `gameJson()` and does not alter gameplay.

If a registered file is renamed/moved within the games folder, refresh environments
to update its location. For moves elsewhere, Open the new path in World Lab to
relink it. Missing files, mismatched IDs, and duplicate live IDs produce errors
and never fall back to another environment. Unresolved course assignments remain
visible until repaired or explicitly reassigned. Corrupt registries are not
silently replaced. Registry discovery validates as a batch before writing.

Legacy world files without IDs load with a new ID and remain unsaved until an
explicit Save persists it. Discovery skips those files; it does not migrate them.
Recovery snapshots retain identity but are not registered as playable locations.
World and registry saves are individually atomic. If a registry write fails
after the world file saves, the editor retains its pre-save state and reports the
partial save; open the saved file and retry Save after fixing the registry.
No existing datasets are migrated.

## Course-driven Quest assignment (selected; integration pending)

One course is one game session. Its ConceptBlock Goals are the assignments
represented by native Quests within that session. The selected structure is:

```text
Curriculum -> Course
                |-> environment_id -> resolve reusable environment
                |-> concept_block_ids -> Goals -> native Quests
                |                                  |
                +-> one shared world state -> GameProgression
                                                   |
                                    per-Quest assignment progress
                                                   |
                                       overall course completion
```

The course owns the environment reference, assignment membership, and overall
session. Each ConceptBlock owns its Goal and executable evidence, supplying one
native Quest. Completing an assignment contributes toward course completion;
changing the selected assignment does not reset world state or erase completed
assignments. Native GameProgression is the planned course-level aggregate above
the individual QuestProgression objects. Assignment completion and course
completion must be displayed separately.

The next authoring feature is **Course -> Assignment -> Play selected** in World
Lab. Selecting a course loads its registered environment and lists only its member
ConceptBlock Goals. Selecting an assignment shows its requirements and binding
diagnostics. Play selected uses the existing single-Goal manual playtest path.
This explicit isolated test evaluates the chosen assignment only; it does not
award course completion or establish a new session per assignment in a course run.
An assignment that depends on earlier world changes may require manual setup
through gameplay; isolated failure is not proof that the course is impossible.

Environment identity, registry resolution, course environment assignment, and the
manual Goal evaluator are implemented. Course-scoped assignment selection and
the shared multi-Quest course runtime remain pending. Missing Goals, unresolved
references, and invalid bindings need explicit diagnostics. An empty environment
reference continues to mean no environment assignment. Selecting another row
must not silently reset a running playtest; starting/resetting a test is explicit.

Course orchestration does not require model-command routing; model-driven play
and feedback insertion remain separate pending work. See the
[next integration phase](WORLD_LAB_GOAL_INTEGRATION_PLAN.md#next-phase-course-driven-assignment-integration-pending).

## Reusable environments and Goal achievability

Manual Goal-driven episodes are implemented. The environment owns entities,
passages, initial facts, and action rules. The database ConceptBlock Goal owns
success criteria, constraints, and their evidence. An episode binds a frozen
Goal snapshot to a fresh interpreter state. Different Goals reuse one compiled
environment; neither evidence attachment nor Goal selection inserts world facts.
See the completed [integration scope](WORLD_LAB_GOAL_INTEGRATION_PLAN.md).

Binding validates predicate support, entity existence, argument types, and arity.
Missing or invalid references reject the candidate without replacing the current
session. An omitted evidence definition is unverified, not a pass. Automatic
whole-Goal success requires at least one success criterion, all success checks
true simultaneously, and executable evidence for every criterion and constraint.
Any true constraint-violation check fails the episode, with failure taking
precedence. Checks are not latched milestones. Sandbox/empty Goals never win by
vacuous truth; Goals with only constraints can fail but cannot automatically win.

Binding validity and compilation do not prove achievability. Manual playtests
remain labelled unknown until successful isolated solution replay supplies a
witness; an already satisfied Goal has a zero-command witness. A failed replay
proves neither impossibility nor general solvability. Only validated, achievable
combinations qualify as verified challenges. A general validator/search that can
prove contradictions or exhaust reachable states remains pending; bounded search
would return unknown rather than unachievable when its budget expires.

Witnesses include environment content, starting-state facts, source ConceptBlock
ID, frozen Goal identity, and action/evaluator semantics. Changes invalidate prior
verification. Full verifier facts and saved solutions are authoring/system data;
future model interaction must use a separate permitted observation projection.

### Episode evidence binding

`scripts/textworld/goal_evaluator.py` supplies the shared typed binder and native Quest adapter
used by startup, reset, commands, and independent replay. Stable world entity IDs
resolve to the compiled entities' typed runtime IDs. The pinned TextWorld 1.7
adapter requires tracked `_facts`; public `facts` substitute display names and
are not used for verification. `carrying(key_1)` checks the corresponding
`in(resolved_key, inventory)` proposition; it does not give the player a key.

The supported predicates are `player_at`, `carrying`, `at`, `in`, `on`, `open`,
`closed`, `locked`, and `eaten`. Unary predicates reject non-empty targets;
binary predicates require correctly typed targets. Passage state checks require
an actual named door. Binding errors report the source ConceptBlock ID,
requirement collection/index, and offending entity when available.

Episode status is `active`, `succeeded`, `failed`, or `stopped`. Reset preserves
the frozen Goal and restarts its initial state. Starting another Goal opens a
fresh interpreter on the same compiled artifact. Database refreshes/edits do not
silently replace an active snapshot. Engine termination and the 1000-command
limit stop an otherwise active episode without asserting Goal success or
inventing a constraint violation. Evaluation results distinguish satisfied,
unsatisfied, and unverified checks; a satisfied constraint check means a
violation occurred.

The adapter constructs TextWorld `Quest`, `Event`, and `QuestProgression` objects.
One win Event contains all success conditions; separate fail Events represent
constraint violations. The progression receives authoritative facts using the
compiled game's knowledge base. Per-requirement display uses native
`Event.is_triggering`. Incomplete evidence disables the win Event while leaving
executable failure Events active. Sandbox/empty Goals create no Quest. Each
episode/reset/replay owns fresh progression. Condition-only Events supply no
action-policy witness or general reachability proof. The semantics identity
`grim-textworld-native-quest-v2` invalidates saved custom-evaluator witnesses.

### Compilation and challenge termination

`build_world()` leaves compiled Quest endings empty. Native Quests are supplied
to the episode's Python `QuestProgression` at startup. Compiled games
contain environment dynamics, while the episode controller owns challenge
termination and rejects commands after any terminal result. Returned `done`,
`won`, `lost`, and challenge `score` reflect native Quest results with coverage
guards, failure precedence, and controller stop limits; raw engine status is
separate under `state.engine`. Goal-only changes reuse compilation. Environment
changes require recompilation; layout, document title, and legacy challenge
metadata do not alter compiled identity.

Mailbox protocol 2 rejects stale clients/workers. Existing workers must restart
and recompile their environment; old binaries with embedded endings are not
loaded into the new session path. Validate world compiles/resets an isolated
candidate and checks binding without replacing live play. Compilation and new
Goal startup preserve the previous session if candidate creation fails.

The first binding implementation requires referenced entities to exist already.
Quest-specific object insertion/setup, role-bound Goal templates, course-driven
episode assignment, model Answer routing, feedback insertion, and
training episode export remain future work. Evidence itself never performs setup.

## Goal selection, manual playtest, and solutions

1. Open or author an environment. Doors, keys, containment, and action inspection
   continue to work as before.
2. Choose **Goal**, then **Reload Goals** to list ConceptBlocks with Goals from
   DataHub's database. Search by ID, name, target, or requirement text (first
   100 matches). Select one to load a snapshot. Edit its requirement text
   and optional condition JSON in DataHub, not in a second World Lab condition
   list. A database load/migration failure blocks selection/import rather than
   overwriting that database. No whole ConceptBlock or database access enters
   the Docker worker: only source identity and Goal JSON are transmitted.
3. For a version-2 world such as `examples/world_lab/locked_door.grimworld.json`,
   choose **Import legacy** explicitly. This creates a ConceptBlock Goal from
   the existing structured conditions and readable deterministic labels. It
   archives the old solution as unverified in a separate episode record, then
   clears legacy metadata in one undoable document edit. Failed persistence
   preserves the original world data. Save the world to persist version 3.
   Legacy conditions are otherwise inactive and never mixed into a selected Goal.
4. **Compile / Start** compiles a changed environment; if it is already current,
   it starts the selected Goal from fresh state without recompiling. Goal view
   shows requirements, binding diagnostics, coverage, and live evaluation results.
   **Validate world** checks a candidate, not general solvability.
5. Choose **Play** and submit one command with Send/Enter. History inspects prior
   observations without rewinding the interpreter; available commands use the
   latest live state. Canvas highlights reflect the selected observation while
   cards still describe the authored initial world.
6. **Reset** restarts the same frozen Goal. **Record** resets and records commands.
   **Save solution** independently replays them through the same evaluator and
   saves a witness only on verified success. Zero-command solutions are supported
   when initial success holds. **Check saved** replays the matching saved witness
   or verifies an imported unverified command list. Stale pairings are rejected.
   Replay never advances or resets the interactive episode.

Episode records live in `models/games/episodes/*.grimepisode.json` (format
`grim-episode`, version 1), separately from environments. Verified records contain
Goal/source snapshot, environment content, commands, and the worker's witness
identity. Imported records retain legacy commands with `verified: false` until
successful replay. Atomic file replacement and bounded reads apply to these
records as to world files. A solution is never an unqualified environment field.

Stop or hiding/leaving World Lab cancels its worker. Requests remain single-flight
with a 120-second timeout. Play supports 1000 commands and replay supports 0-1000.
Playable names remain unique ASCII names of at most 80 characters, starting with
a letter. TextWorld source delimiters in names/descriptions are rejected.
Containers/supporters must be directly in rooms; portable objects may be nested.
The worker supports 1000 entities and 2000 passages.

## Worker requirements and transport

Use the already installed Docker Desktop Linux engine and `grim-textworld:1.7.0`
image. On Windows and macOS, the client searches PATH and Docker Desktop's
standard installation path.
It finds `scripts/textworld/world_worker.py` by walking ancestors of the working
directory and executable. No image rebuild is needed: the worker source is mounted
read-only. The client never pulls/builds an image or installs dependencies.

`ui/world_lab/textworld_client.*` launches a hidden Docker process asynchronously
with networking disabled. It mounts only the worker directory read-only and an
exclusively created temporary session directory writable. Requests/responses are
protocol-2, ID-matched JSON files published by rename, limited to 16 MB. This file
mailbox keeps compiler stdout/stderr out of the protocol and avoids blocking pipe
writes on the UI thread. The worker keeps one interpreter alive between requests.
Temporary games and logs are removed with the owned session on shutdown. The
worker also exits after 15 minutes without requests. The native client supports
Windows and POSIX hosts, including macOS.

## Editing

- A new document starts with one room and its player-start marker.
- Drag a room's heading to reposition its card. Pan with the middle button and
  zoom with the wheel. Fit frames all rooms.
- Drag a compass handle onto another room to connect it. Drop on empty canvas
  to create a room and its connection as one undoable operation. Connections are
  reciprocal, occupy both compass exits, and retain their directions when cards move.
- Select a room, container, or supporter, choose an object kind, and use Add here.
  Objects, keys, containers, supporters, and food are supported. Nested contents
  appear as indented rows inside the containing room.
- Drag an object row onto a room heading, container, or supporter to relocate it.
  The inspector's Location selector also provides valid destinations.
- Edit names and descriptions in the inspector; use Apply changes or Enter.
  Selecting another entity, saving, leaving the tab, and periodic recovery also
  commit valid pending edits. Invalid edits remain in the inspector with an error.
- Containers have an open/closed starting state. Select a room and choose Set
  player start to change the spawn room.
- Select a connection by clicking its line. Delete selection removes that passage.
  Deleting a room or container also removes its contents and incident connections;
  deleting the spawn room moves the spawn to the first remaining room.
- Undo/Redo cover document changes, including New/Open. A completed drag is one
  operation; cancelled drags do not change the document. Ctrl+Z, Ctrl+Shift+Z,
  Ctrl+Y, Delete, and Escape work while the canvas has keyboard focus. Text fields
  retain their own keyboard handling. History holds up to 100 edits.
- Scroll the inspector if the panel is too short to show all its controls.

## Files and recovery

Games default to `resources/models/games` under the resolved resource directory.
The folder is created automatically. New chooses an unused `world.grimworld.json`,
`world-2.grimworld.json`, etc. Rename the filename in the path field before saving
to give a game a recognizable name. The Saved games dropdown opens direct
`*.grimworld.json` children and refreshes after saves and every two seconds while
closed. Copy existing games into this folder to include them. Open/Save still
accept custom file paths, whose parent directory must already exist. Unicode paths are supported. Save writes into a
unique staging directory beside the destination, closes the complete file, then
atomically replaces the destination on Windows. Failed saves preserve the previous
file and do not mark the document saved. Malformed or unsupported files cannot
replace the current document.

Valid document changes are checkpointed every ten seconds while the tab is active
and when leaving the tab. On Windows the recovery file is
`%LOCALAPPDATA%/GRIM/WorldLab/recovery.grimworld.json`. Recover explicitly opens that
snapshot; it does not overwrite the chosen world file. Save after recovering.
An untouched new session does not replace a previous recovery snapshot. Recovery
is a single per-user snapshot, not a multi-process document library.

## Implementation

- `ui/world_lab/world_document.*`: renderer-independent document, validation,
  transactional snapshot history, persistence.
- `ui/world_lab/ui_world_canvas.*`: room and object presentation, hit testing,
  drag previews, compass connections, selection, canvas shortcuts.
- `ui/world_lab/ui_world_lab_view.*`: inspector, toolbar, recovery and lifecycle.
- `ui/ui_training_panel.*`: World Lab registration and Goal database callbacks.
- `ui/world_lab/world_goal.hpp`: Goal selection boundary, legacy conversion,
  bounded versioned episode-record reads, and safe record filenames.
- `ui/world_lab/ui_world_session.*`: selection, immutable snapshots, Play lifecycle,
  separate witness persistence, and progress display.
- `scripts/textworld/goal_evaluator.py`: typed binding, native Quest evaluation, and episode
  status; bump its semantics identity when action/check behavior changes.
- `scripts/textworld/world_worker.py`: questless compilation, typed-state adapter,
  transactional episode startup, command handling, and isolated replay.

The JSON format is identified by `format: "grim-world"`. New environment-only
files save as version 3. Version-1 and version-2 files remain readable; pending
legacy challenge data saves as version 2 until explicitly converted, preventing
silent loss. Version-3 files reject challenge/solution fields.
Entities and
passages use stable IDs; `spawn` references a room. Parent references represent
containment. `editor.rooms` holds layout independently of passage directions and
entity state. Validation rejects duplicate IDs, dangling references, invalid
holders, containment cycles, conflicting exits, missing layouts, and nonfinite or
out-of-range positions. Files are limited to 16 MB, 10,000 entities and 20,000
passages. This is structural validation, not proof that a puzzle is solvable.

Version 2 introduced door fields and embedded challenge/solution metadata.
Version 3 retains door fields but separates challenge data into ConceptBlock
Goals and episode records. Deleting entities still cleans pending legacy
references atomically. Model episodes and curriculum generation remain pending;
importing arbitrary TextWorld games is not supported.

## Validation

Run the host-only tests with the existing MSVC toolchain:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -WorldLab
```

The suite covers Goal selection/import, legacy preservation, episode-record
persistence, round-trip and replacement saves, failed loads/saves, invalid
worlds, cascading deletion, history, cancelled gestures, room creation through
ports, object relocation, inspector creation/type selection, recovery, and tab
lifecycle. Rendering uses test sinks. Native visual QA requires a separately
authorized application build/run.

Host binding tests need no TextWorld installation; native evaluation cases skip
when it is absent. Run the full evaluation suite in the existing image:

```powershell
python -m unittest discover -s tests -p test_world_goal_evaluator.py -v
docker run --rm --pull never --network none --entrypoint python --mount 'type=bind,source=D:/G.R.I.M/scripts/textworld,target=/worker,readonly' --mount 'type=bind,source=D:/G.R.I.M/tests,target=/tests,readonly' -e PYTHONPATH=/worker grim-textworld:1.7.0 -m unittest discover -s /tests -p test_world_goal_evaluator.py -v
```

For real Docker compiler/interpreter and native transport integration, using only
the existing image:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -WorldLab -TextWorld
& 'C:/Program Files/Docker/Docker/resources/bin/docker.exe' run --rm --pull never --network none --mount 'type=bind,source=D:/G.R.I.M/scripts/textworld,target=/worker,readonly' --mount 'type=bind,source=D:/G.R.I.M/tests,target=/tests,readonly' --mount 'type=bind,source=D:/G.R.I.M/ui/world_lab,target=/catalog,readonly' grim-textworld:1.7.0 python /tests/test_world_worker.py
```

These verify locked-door gameplay, win/failure, reset, independent solution replay,
failed-compilation preservation, single-flight transport, cancellation/restart,
version migration, and challenge reference cleanup. No `grim.exe` build/run is
part of either test command.

## Built-in action inspection

World and Asset inspectors provide an Actions shortcut. Rules and command
templates come from the generated TextWorld 1.7 catalog, while native code only
binds typed parameters to document entity IDs. World inspection can prepare a
command in Play. Play also offers an Actions composer and a live available-command
picker; neither submits until Send/Enter. The worker returns its catalog during
compilation to detect a stale offline snapshot. Definitions remain read-only.

Regenerate the snapshot only when intentionally updating the installed knowledge
base, then run the host and worker tests:

```powershell
& 'C:/Program Files/Docker/Docker/resources/bin/docker.exe' run --rm --pull never --network none --mount 'type=bind,source=D:/G.R.I.M/scripts/textworld,target=/worker,readonly' --mount 'type=bind,source=D:/G.R.I.M/ui/world_lab,target=/output' grim-textworld:1.7.0 python /worker/action_catalog.py --header /output/textworld_action_catalog.generated.hpp
```

The snapshot contains command parameters and logical preconditions/additions/
removals, not a duplicate execution implementation. Custom actions, container
key configuration are not part of this slice. Path traversal is a generic
World Lab extension described below; the bundled default TextWorld catalog
remains unchanged.

## Path navigation primitive

Select a room in Inspect and choose **Navigation (world-wide) → Path navigation**.
The selected room becomes `/`. Remove existing compass passages first; the mode
switch never silently discards them. Other rooms initially receive their stable
IDs as child segments. New rooms are also added beneath the root. Edit **Room
path** and apply to rename or reparent a room beneath an existing address. Display
names and object containment are independent of navigation addresses. Renaming
a parent updates descendant paths while preserving entity IDs. Deleting a room
deletes its navigation subtree and contained objects; Undo restores it.

Path worlds save as `grim-world` version 4, with a separate navigation tree:

```json
"navigation": {
  "mode": "path",
  "rooms": [
    {"id": "r1", "parent": "", "segment": ""},
    {"id": "r2", "parent": "r1", "segment": "section"}
  ]
}
```

Every room must occur exactly once; exactly one root has an empty segment.
Nonroot segments contain 1–80 ASCII letters, digits, underscores or hyphens.
Paths are case-sensitive, unique, and limited to 509 bytes (leaving room for
`go ` in the command limit). Missing parents, cycles, and compass passages are
rejected. Goals and episode recordings remain separate from version-4 worlds.
Older version-1/2/3 compass files retain their existing semantics. Switching
back to Compass removes address metadata in one undoable edit.

Commands:

- `enter section`: enter a direct child using its segment.
- `go /section`: use an absolute address.
- `go section`, `go ./section`: use a relative address.
- `go ..` or `back`: move to the parent, rather than the previously visited room.
- `go /`: return to the root when it is the directly connected parent.

Each command crosses exactly one parent/child edge. Absolute paths do not
teleport, and paths never refer to host filesystem locations. Unknown addresses,
nonadjacent destinations, and traversal above `/` return errors without changing
the session. `look` retains TextWorld inspection behavior. Observations add
`location_path` and adjacent `available_rooms`; feedback prints the current path,
and the live command picker offers path commands. The offline composer offers
`go <room path>` and `back` instead of compass moves.

`scripts/textworld/path_navigation.py` extends a private TextWorld knowledge base
with `path_link` facts and a `go/path` action, and compiles matching Inform7
movement. TextWorld updates its own `at(P, room)` fact and move count. Live play
and witness replay use the same command resolver; the navigation tree and its
semantics participate in environment identity. No subsystem bindings or specific
environment content are part of this primitive. Room assets remain individual
room/object trees, without external navigation links.

Focused address validation: `python -m unittest discover -s tests -p test_path_navigation.py -v`.
The existing worker suite also covers native path movement, branching, reset,
Goal evidence and replay. Run the host UI suite for authoring integration.

## Reusable assets

Training Control now includes an **Assets** tab. The library lives under
`resources/models/world_assets/rooms` and `resources/models/world_assets/objects`.
Each version-1 `.grimasset.json` contains a library name, root entity ID, and a
self-contained entity tree. It does not contain passages, player start, goals,
solutions, or references to other games. A room includes all contained objects;
a container or supporter includes all its descendants.

- In World Lab, select a room or object and use **Save as asset** in Inspect
  (scroll down if necessary). Assets opens a separate editable copy; press Save.
- To start from scratch, choose a type in the inspector and click **New asset**.
- Edit the library name at the top. Select an item in the canvas to edit its
  in-game name, description, or container state. Choose an object type and
  **Add inside** to add contents to the selected room/container/supporter.
- **Save** updates the current asset; **Save copy** writes a separate file.
  Changing the library name updates its browser label without moving its file.
- **Saved assets** lists both categories and refreshes while its menu is closed.
  Malformed files are labelled invalid; attempting to open one preserves the editor.
- **Add to world** inserts into the currently open World Lab game and switches
  back to World Lab. Rooms appear in the canvas center. For objects, first select
  a room, container, or supporter in World Lab. Each insertion is one undoable
  operation, remaps every entity ID and parent, and adds numeric suffixes to
  duplicate names. Save the game to persist the inserted copy.
- **Undo/Redo** cover asset edits. **Discard edits** restores the last loaded or
  saved snapshot. Unsaved edits block New, opening another asset, capture from
  World Lab, and deletion until saved or explicitly discarded. Switching tabs
  preserves the draft in memory; asset drafts are not autosaved across app exits.
- **Delete asset** moves a saved asset into `world_assets/.trash`. Restore its
  file to `rooms` or `objects` to recover it. Existing games retain their copies.

The preview canvas supports selection, pan and zoom. Room topology and the
synthetic preview room for standalone objects cannot be edited in this tab.
Asset copies are independent; linked updates and multi-room templates are not
implemented. TextWorld still validates playable names and nesting at compilation.

The host-only UI test command above also covers asset round trips, invalid
references, repeated insertion, unique names/IDs, failed placement rollback,
atomic undo, saving, library selection, dirty-edit protection and recoverable deletion.
