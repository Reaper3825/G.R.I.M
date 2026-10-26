# World Lab — Stages 1 and 2

For the full control reference, customization guide, terminology index, and agent
maintenance notes, see [World Lab and Assets reference](WORLD_LAB_REFERENCE.md).

World Lab is an authoring tab in the Training Control panel. It stores an editable
world locally. Validate and Compile & Play start an isolated TextWorld Docker
worker on demand. No model, inference service, or training process is started.

Courses store an optional environment reference as `Course::environment_id`,
persisted in each course entry in `curriculum_registry.json`. An empty string
means no environment is assigned; older registries without the field load with
that default. World Lab documents do not yet have persistent game IDs, and
course-to-world resolution and an environment picker remain pending. Document
entity IDs identify entities within a world; worker `game-N` filenames are
temporary session artifacts and must not be used as course environment IDs.

## Reusable environments and Goal achievability (selected design; pending)

A reusable environment owns the subjects/entities, map, initial state, and
permitted action rules. It remains stateful during play. The selected Goal owns
success checks and constraint-violation checks. Separating the Goal from the
environment does not make gameplay stateless. Different ConceptBlocks may use
the same environment with different Goals and fresh episode states.

Before starting a challenge, validate the combination of environment and Goal:

- Resolve every evidence subject and optional target in the selected environment;
  validate predicate support and argument types. Missing entities or unsupported
  checks make the combination invalid.
- Check the success conditions together, including obvious contradictions such
  as requiring the same object in two exclusive locations. Check for conflicts
  with failure conditions. Individually possible criteria need not be jointly
  achievable.
- From the actual episode starting state, establish a permitted action sequence
  that reaches a state satisfying every success condition without triggering any
  constraint-violation condition along the way or at the final state. A required
  key existing somewhere in the world does not prove it is obtainable.
- Evaluate initial success/failure as well as subsequent states, with failure
  taking precedence. Report an already satisfied Goal separately from a challenge
  requiring actions; an initial violation is a failed starting state.

Validation must distinguish `invalid`, `achievable`, `unachievable`, and
`unknown`. A verified successful replay is a witness for `achievable`. A bounded
search that finds no solution returns `unknown` unless it exhausts the complete
reachable state space; contradictions or an exhaustive search can establish
`unachievable`. Do not label compilation, type validation, a search timeout, or
the model's claim as proof of achievability or impossibility. Only validated,
achievable combinations qualify as verified challenges.

Run reachability checks on an isolated state, using the same action transitions
and evidence evaluator as gameplay. Associate any witness or cached result with
the environment content/revision, episode starting state, Goal evidence, and
action/evaluator semantics; changes invalidate the result. Verifier access to
full world state and solution witnesses does not authorize exposing them to the
model. Goals containing requirements without executable evidence cannot receive
an automatic whole-Goal achievability guarantee.

The reusable-game path is intended to evaluate Goal evidence in the controller
after each transition, without embedding each Goal's endings into a separate
compiled game. That controller and the achievability validator are pending.
Current manual Play still compiles the document's Quest into the game, and
Validate checks compilation/reset rather than general solvability.

### Episode evidence binding (selected design; pending)

The Goal storage prerequisite is implemented: ConceptBlock criteria and
constraints can persist optional structured evidence, and the model renderer
excludes that evidence. See the
[Goal evidence contract and migration](GRIM_REASONING_DATA_CONTRACT.md#44-evidence--system-facing-executable-condition).
This does not yet attach those Goals to manual Play; the binding, evaluator,
compilation, and episode changes below remain pending.

The environment definition owns a stable identity/revision, entities, passages,
initial state, and action rules. The ConceptBlock Goal owns the target text,
success criteria, constraints, and their structured evidence. An episode owns
the selected environment instance, a frozen Goal snapshot with source identity,
bound evidence checks, command history, evaluation results, and terminal status.
Editing a ConceptBlock must not change a running episode's Goal.

Attaching a Goal registers checks against the environment; it does not insert
the requested conditions as true world facts or alter the starting state.
For example, binding `{"predicate":"carrying","subject":"key_1"}` resolves
the stable entity ID `key_1` and prepares a check for TextWorld's
`in(resolved_key, inventory)` proposition. It does not give the player the key.
Display names are not binding keys. Evidence remains paired with its criterion
or constraint; current truth values and verification diagnostics belong to the
episode, separately from those definitions.

Episode creation and execution follow this lifecycle:

1. Load the selected environment and its stable-ID-to-runtime-entity mapping.
2. Load and freeze the selected ConceptBlock Goal and its source identity.
3. Bind each evidence `predicate`, `subject`, and optional `target` against that
   mapping. Validate predicate support, entity existence, and argument types;
   reject invalid bindings before challenge execution. Do not derive executable
   conditions from requirement prose or treat missing evidence as a pass.
4. Perform the combination/achievability validation described above on an
   isolated state. Keep bound success and violation checks in the episode.
5. Evaluate the initial state and the authoritative state after every command.
   Any constraint violation fails the episode; otherwise, all success checks
   must hold simultaneously for automatic success. Requirements without an
   applicable verifier prevent an automatic whole-Goal success guarantee.

Use shared predicate definitions and binding/evaluation semantics for authoring
validation, live play, reachability checks, and saved-solution replay. The
existing worker's condition-to-proposition conversion is the starting point for
that shared binding layer. The reusable path compiles the environment without
Goal-specific Quest endings and attaches checks to an episode-owned evaluator.
Changing only the Goal may reuse the compiled environment, with a fresh episode
state. The controller's evaluated Goal status is authoritative for these
challenges; quest-specific TextWorld `won`/`lost` fields are not its completion
signal. Engine termination and episode limits must still be handled separately.
Verifier access to authoritative facts does not make those facts model-visible.

### Compilation and challenge termination transition (pending)

The Goal is the source of the quest's win/fail conditions:
`success_criteria[].evidence` supplies success checks, and
`constraints[].evidence` supplies constraint-violation checks. Their paired
`criterion` and `constraint` text supplies the model-facing requirements. The
reusable environment must not maintain a second independent set of challenge
win/fail conditions.

Currently, `build_world()` assigns document conditions to `maker.quests` before
`maker.compile()` creates the playable `.z8` file. Those compiled Quest events
can end the interpreter's game when their conditions fire. The reusable path
must remove that Goal-specific Quest construction from environment compilation;
merely attaching a new Goal evaluator to an old compiled game would leave the
old endings active.

Recompile existing games used by the reusable path to remove those embedded
challenge endings, and invalidate cached artifacts that still contain them.
Thereafter, changing only the selected Goal binds new episode checks without
recompiling an otherwise unchanged environment. Environment or action-rule
changes can still require recompilation.

The episode controller evaluates the selected Goal at reset and after each
command, marks the challenge failed on any constraint violation or successful
when all success checks hold, and rejects further gameplay commands once the
episode is terminal until a reset or new episode. Saved-solution replay uses
the same checks and termination rules. Native engine termination or an episode
limit does not by itself establish Goal success. These are implementation
requirements for the selected design; current manual Play is unchanged.

Saved solution witnesses belong to the environment-and-Goal pairing, including
the starting state and semantics revisions used for verification, rather than
to the environment alone. Revalidate them when those inputs change.

The first binding implementation requires referenced entities to exist in the
authored environment. If future quests introduce objects or alter initial
placements, represent that as an explicit episode setup operation, apply it
before evidence binding, and include its resulting starting state in validation
and cache identity. Evidence itself never performs setup. Parameterized Goal
templates and role-to-entity bindings are future extensions, not prerequisites
for reusing one environment across different Goals.

Future model-driven play follows the
[recurrent reasoning contract](GRIM_REASONING_DATA_CONTRACT.md): observe DDE,
submit only the completed Answer command, then supply actual environment feedback
before the next cycle. Define and Execute are optional; ordinary responses do
not require tool calls. An answer boundary ends a response cycle, not necessarily
the challenge. Model dispatch and feedback insertion remain pending; see the
[integration reference](WORLD_LAB_REFERENCE.md#recurrent-model-interaction-contract-selected-integration-pending).

## Goal and manual playtest

1. Open `examples/world_lab/locked_door.grimworld.json` using the file field for
   a complete two-room puzzle, or author a world in the canvas.
2. Select a passage and use Inspect. Enter a door name, choose open/closed/locked,
   select its matching key, and Apply changes. A blank name means an ordinary
   passage. A locked door must have a key. Deleting that key clears the reference
   and changes the door to closed; undo restores both.
3. Choose Goal. Enter the objective and add conditions using the predicate,
   subject, and (where needed) target controls. **Every win condition must hold
   together; any fail condition ends the challenge in failure.** These are state
   conditions, not ordered milestones. Select a saved condition and Remove to
   delete it. Supported conditions cover player location, carrying, object
   location, container/door state, and eaten food.
4. Validate compiles a temporary game and resets its interpreter. It checks
   TextWorld constraints and compiler compatibility, not general solvability.
   Errors identify an editor entity when possible. With no win conditions the
   world is an exploration sandbox.
5. Choose Play, then Compile & Play. Enter commands and press Enter or Send.
   History shows each response, inventory, and available commands as an authoring
   aid. Selecting a history entry inspects that observation; it does not rewind
   the live interpreter. The next command always acts on the latest live state.
   The current room is outlined in green; changed objects are highlighted in
   amber. Card placement and object lists still show the authored starting state.
6. Reset returns the compiled game to its starting state. Record resets first and
   records subsequent commands. Save solution replays those commands in a separate
   interpreter and only saves them into the document if they reach a win.
   Check saved replays the document's saved solution against the compiled game.
   Save the world file to persist the solution. Verification never advances or
   resets the interactive session.

Semantic edits require a fresh Compile & Play before commands, reset, or solution
verification. Layout-only moves and solution metadata do not invalidate the game.
Stop cancels an in-flight operation and removes the owned worker. Leaving/hiding
the World Lab tab also stops it; reopening requires compilation again. Requests
have a 120-second timeout and only one can run at a time. Episodes/solutions are
limited to 1000 commands.

TextWorld's supported authoring subset has tighter constraints than the document:
playable names are unique (case-insensitive), begin with a letter, and contain
ASCII letters, digits, spaces, and hyphens (80 characters maximum). Quotes,
brackets, backslashes, and line breaks in names/descriptions/objectives are rejected
because TextWorld embeds them into Inform source. Containers and supporters must
be directly in rooms; portable objects may be inside containers or on supporters.
The worker accepts up to 1000 entities and 2000 passages. These limits produce
actionable validation errors rather than silently rewriting an authored world.

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
versioned, ID-matched JSON files published by rename, limited to 16 MB. This file
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
- `ui/ui_training_panel.*`: World Lab tab registration and dispatch.

The version-2 JSON format is identified by `format: "grim-world"`. Version-1 files
load with empty challenges/solutions and doorless passages; saves use version 2.
Entities and
passages use stable IDs; `spawn` references a room. Parent references represent
containment. `editor.rooms` holds layout independently of passage directions and
entity state. Validation rejects duplicate IDs, dangling references, invalid
holders, containment cycles, conflicting exits, missing layouts, and nonfinite or
out-of-range positions. Files are limited to 16 MB, 10,000 entities and 20,000
passages. This is structural validation, not proof that a puzzle is solvable.

Version 2 adds door name/state/key fields, objective text, win/fail condition
arrays, and a recorded solution. Deleting entities removes dependent conditions
in the same undo transaction. Model episodes and curriculum generation remain
Stage 3. Importing arbitrary TextWorld games is not supported.

## Validation

Run the host-only tests with the existing MSVC toolchain:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/run_ui_input_foundation.ps1 -WorldLab
```

The suite covers round-trip and replacement saves, failed loads/saves, invalid
worlds, cascading deletion, history, cancelled gestures, room creation through
ports, object relocation, inspector creation/type selection, recovery, and tab
lifecycle. Rendering uses test sinks. Native visual QA requires a separately
authorized application build/run.

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
key configuration, and non-compass traversal are not part of this first slice.

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
