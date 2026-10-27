# Goal-driven TextWorld episodes — implementation scope

Status: implemented for manual Play and independent solution replay. Goal binding,
controller termination, protocol-2 worker operations, database selection/import,
version-3 environment documents, and separate version-1 episode records are in
place. The [World Lab guide](WORLD_LAB.md) documents current usage. Model routing,
feedback insertion, general reachability proof, and course-driven episode
assignment remain future work. Persistent environment identity, registry
resolution, and course environment assignment are implemented as the next-phase
foundation described below.

The sections below retain the implementation boundaries and acceptance criteria
for maintenance; they do not imply those manual integration components are still
pending. Verified replay provides achievability witnesses, while unproven manual
playtests remain unknown. Existing production datasets are not automatically
migrated or overwritten by this work.

## Outcome and boundary

Compile one environment, select a ConceptBlock Goal from the database, and run a
manual episode whose outcome is determined by that Goal's evidence. Starting
another Goal creates fresh episode state using the same compiled environment.
Goal changes do not rebuild the game or mutate the authored environment.

Include the binder, evaluator, worker lifecycle, native session adapter, minimal
Goal selection/status UI, and independent solution replay. Deliver these as one
usable path rather than removing compiled endings before their replacement can
enforce completion. No model-command dispatch, inference feedback insertion,
training export, general reachability solver, quest-specific object spawning,
parameterized Goal templates, or environment/course registry resolution is part
of this phase. Existing datasets are migrated explicitly with the storage-phase
tooling; the integration must not rewrite them as a side effect of selection.

The first environment selector is the currently open World Lab document. Use an
environment-content fingerprint and starting-state identity for episode/replay
compatibility; temporary worker filenames are not persistent world IDs. A future
environment registry can replace this selection mechanism without changing the
binder/evaluator interface.

## Component boundaries

| Component | Responsibility |
|---|---|
| Native Goal selection adapter | Read the selected ConceptBlock by ID, validate its Goal JSON, and send only a Goal snapshot plus source identity to the worker. Do not send reasoning phases, answer text, or whole database access into Docker. |
| Environment builder | Build entities, passages, initial facts, action rules, and an authoritative stable-ID/runtime-symbol map. Compile without challenge-specific Quest endings. |
| Evidence binder | Resolve and type-check conditions once, retaining requirement collection/index for diagnostics. Return immutable bound checks without modifying world facts. |
| Evidence evaluator | Supply bound conditions to native TextWorld Quest/Event/QuestProgression using authoritative facts and the game's knowledge base. Return current Event checks and native progression outcomes with coverage guards and failure precedence. |
| Worker episode controller | Own a frozen Goal, bindings, fresh interpreter state, history, and terminal status. Evaluate on reset and after each command; prevent commands after termination. |
| Native Play view | Display the selected Goal and progress, and consume controller status. Keep inspection history separate from live state. |
| Solution verifier | Replay in an independent interpreter with the same Goal, binder, evaluator, and termination rules; retain pairing metadata with verified commands. |

Keep predicate semantics in one worker-side module, separate from file-mailbox
transport and interpreter lifecycle. Binding remains testable without TextWorld;
native evaluation tests run in the existing TextWorld image. The adapter creates
one win Event containing all success conditions and separate fail Events for
constraints. It updates episode-owned QuestProgression from authoritative facts,
and uses Event.is_triggering for current per-requirement display. Missing evidence
disables winning, while executable failures remain active. Empty/sandbox Goals
create no Quest. Compiled endings remain empty; Goal replacement reuses the
artifact. The native-quest-v2 semantics identity invalidates earlier witnesses.
Native authoring controls
may filter choices, but worker validation remains authoritative.

## Implementation sequence

### 1. Extract predicate binding and evaluation

Add focused modules under `scripts/textworld/` for predicate definitions,
binding, and evaluation. Initially support the existing predicates:
`player_at`, `carrying`, `at`, `in`, `on`, `open`, `closed`, `locked`, and `eaten`.

Validate supported predicates, subject/target existence, argument types, and
arity. Reject non-empty targets for unary predicates and missing targets for
binary predicates. State checks on a passage require an actual door. Do not
treat normalized display names as authoritative entity identities: preserve the
typed runtime identity from the compiled game's mapping, including player and
inventory symbols. Verify that identity against actual returned TextWorld facts.

Return diagnostics with ConceptBlock ID, requirement collection/index, and
offending entity when available. Report malformed evidence as invalid. Report
an omitted evidence definition as unverified, never satisfied.

### 2. Introduce explicit episode state

Represent the active episode separately from the compiled environment. Retain
environment fingerprint, starting-state identity, ConceptBlock source ID, frozen
Goal/evidence identity, bindings, history, and evaluator-semantics version.

Distinguish requirement evaluation from episode termination. Each check can be
satisfied, unsatisfied, or unverified. The episode can be active, succeeded,
failed, or stopped without verified success (for example, an interpreter ending
or a command limit). Binding errors reject the candidate episode before it
replaces a valid current session.

Evaluate the initial state before accepting a command, then every post-command
state. Any true constraint-violation condition fails the episode, even when all
success conditions are true. Success requires at least one success criterion,
all success evidence true simultaneously, and every applicable requirement
covered by valid executable evidence. Missing evidence on a criterion or a
constraint prevents an automatic whole-Goal success claim. An empty Goal or no
selected Goal permits sandbox exploration and must not win by vacuous truth.
Requirements are state checks, not permanently latched milestones.

Reset restarts the same frozen Goal from its starting state. Selecting another
Goal starts a fresh episode; switching Goals never reuses the old live state.
Database edits do not change an active Goal. Engine termination and limits are
handled explicitly rather than becoming success or a fabricated constraint
violation.

### 3. Replace embedded endings and separate worker operations

Change `scripts/textworld/world_worker.py` so environment compilation omits the
current Goal-specific `maker.quests` construction. Compile/open/reset a questless
candidate in the existing TextWorld 1.7 image and prove that it remains playable;
do not assume how native questless `done`, `score`, `won`, or `lost` behave.

Separate compile/validate-environment operations from start/reset/step-episode
operations. The episode-start request carries source identity and a Goal snapshot
separately from the environment. Return explicit episode status and requirement
results. Preserve raw compiled-engine status separately for diagnostics; native
Python Quest progression supplies challenge outcomes. Version the mailbox protocol
for this semantic change and update the client and worker together.

Invalidate old compiled snapshots that contain embedded challenge endings and
require recompilation before reuse. Keep candidate compilation and episode
startup transactional: failures must preserve the previous playable session.
Goal-only changes retain the compiled artifact. Preserve request IDs, size
limits, single-flight execution, timeouts, cancellation, and worker isolation.
Full verifier facts/results remain separate from any future model-visible
observation projection.

### 4. Connect database selection and manual Play

Add a minimal ConceptBlock selector to World Lab using the existing database
access layer. Snapshot by stable ConceptBlock ID; show target text, criterion and
constraint text, binding errors, coverage, and current verification results.
Do not create a second editable win/fail list alongside the selected Goal.

Split native compatibility checks: environment changes invalidate compilation;
Goal changes invalidate the active episode and saved verification for that Goal.
Layout changes invalidate neither. Update Play completion, Reset, Record, and
history handling to consume episode-controller status.

Old version-2 world files retain their structured win/fail conditions only as
legacy import input. Provide an explicit conversion that creates a ConceptBlock
Goal using those existing conditions as evidence, with deterministic readable
requirement text; do not infer checks from objective prose. Keep the original
file intact until conversion/save succeeds. The reusable world representation
then excludes challenge conditions and solution commands. Preserve converted
challenge/solution data in separate Goal/assignment records, with a versioned
serialization and rollback-safe saves. A database-selected Goal must never be
combined implicitly with legacy conditions.

### 5. Move solution verification onto the same evaluator

Replay from the exact starting state in an isolated interpreter, using the
frozen Goal and the same reset/step checks. Support a zero-command witness when
initial success is already verified. Reject commands after any terminal result.
Verification must not advance, reset, or replace the interactive episode.

Associate each witness with environment content, starting state, Goal/evidence
identity, and action/evaluator semantics. Changes invalidate prior verification.
Do not retain a solution as an unqualified property of a reusable environment.

## Validation and acceptance

Use host Python tests for binding, existing-image tests for native evaluation, and focused native host tests for
snapshot transport, selection, persistence, and UI/session lifecycle. Extend
`tests/test_world_worker.py` for real compilation/interpreter checks using only
the existing Docker image. No application/trainer/model-server build or run,
package installation, or image rebuild is required by this scope.

The defining integration test compiles one environment once, starts Goal A,
reaches its outcome, then starts Goal B on a fresh state using the same compiled
artifact. Goal A's conditions must not terminate Goal B. Assert unchanged
artifact identity and compile count, not merely matching filenames.

Also cover every supported predicate and invalid type/reference/arity; doors
versus doorless passages; an ineffective command; simultaneous success checks;
any-failure behavior; initial success/failure; failure precedence; missing
evidence; empty/sandbox Goals; rejected post-terminal commands; reset; frozen
snapshots; failed candidate preservation; stale witness rejection; independent
replay; and legacy import preservation. Confirm that attaching evidence never
changes initial world facts.

Binding validity is not achievability. This phase may report achievable only
when a replay supplies a verified witness. Without a witness or proof, report
achievability as unknown and permit clearly labelled manual playtesting, not a
verified challenge assignment. An initially satisfied, fully verified Goal has
a zero-command witness. A general search/proof validator and its challenge
admission policy remain separate work; compilation or a failed replay proves
neither general solvability nor impossibility.

Completion means manual Play and replay use the database Goal as their sole
challenge authority, the same compiled environment supports multiple Goals, old
embedded endings cannot interfere, and documentation distinguishes the completed
manual integration from pending model-driven interaction and reachability work.

## Next phase: course-driven assignment (integration pending)

Environment foundation status: implemented. World documents persist independent
IDs; `environment_registry.json` maps IDs to validated files; World Lab saves/opens
register locations; DataHub offers course Environment assignment and discovery
refresh. Explicit Save persists legacy IDs; Save copy creates a new identity.
Missing files, mismatches, duplicates, and malformed registries produce errors.
See [assignment and resolution](WORLD_LAB.md#assign-and-resolve-an-environment).
The remaining work separates selected-assignment authoring from the overall
course game session.

### Session ownership

- One Course is one game session, using its resolved environment and shared world state.
- Each member ConceptBlock Goal is one assignment, represented by one native Quest.
- Assignment completion contributes toward overall course completion. Selecting
  another assignment does not reset world state or discard completed assignments.
- Native GameProgression is the planned course aggregate over individual
  QuestProgression objects. Goals remain distinct Quests within that game.

### Next deliverable: play a chosen course assignment in World Lab

1. Add a course selector backed by the existing course registry. Resolve the
   chosen course's environment ID through EnvironmentRegistry and load that world.
2. List member ConceptBlocks by stable ID and name. Show missing blocks and
   blocks without Goals as explicit diagnostics rather than silently omitting them.
   Do not substitute the global Goal database search for course membership.
3. Selecting a member previews its target, criteria, constraints, and binding
   diagnostics. Keep source course ID, ConceptBlock ID, and frozen Goal snapshot.
4. An explicit Play selected action starts the existing isolated single-Goal
   manual test in the course environment. Only the chosen assignment's Quest
   determines its test result. Reset test restarts that test; selecting another
   row alone must not reset or replace live play. Goal-only tests reuse compilation.
5. Label the result as assignment playtest success/failure, separate from overall
   course completion. Save course/assignment provenance with any new test records;
   existing manual records remain readable and are not retroactively course credit.
6. Preserve the current document and running test if course/environment loading,
   Goal lookup, or binding fails. Pending editor changes need explicit handling
   before loading a different world; do not silently discard them.

Selected-assignment playtesting is an authoring capability, not the course runtime
session boundary. It may begin from the saved environment's initial state and
allow manual gameplay setup. An assignment depending on earlier assignments may
not be achievable in isolation; report the result without declaring the course
impossible or inventing prerequisite state. Automatically creating entities or
injecting success facts is outside this feature.

Acceptance: select one course with two member Goals, load its registered world,
choose each Goal and explicitly playtest it, confirm native Quest evaluation and
unchanged compiled artifact/build count for Goal-only changes. Verify membership
filtering, missing blocks/Goals, unresolved worlds, invalid bindings, frozen
snapshots, failed-load preservation, and distinct test-result provenance.

### Subsequent deliverable: shared course game session

Load the course's assignment Quests into one game session and aggregate their
progress with native GameProgression. World changes and assignment completions
persist across assignment focus changes. Completing one Quest must not terminate
the game while overall course requirements remain incomplete. Reset course is a
separate operation from an isolated assignment-test reset. Course requirement
coverage and failure handling must remain explicit; missing evidence cannot
produce a verified course-completion claim.

Acceptance: complete two assignments in the same interpreter session, prove that
world changes survive switching focus, and show assignment progress separately
from course completion. The course must remain active after only the first
required assignment completes.

Neither deliverable requires model commands, inference feedback, training export,
or a general reachability solver. The native GameProgression adapter and its
aggregate semantics need focused validation against the pinned TextWorld version
when the shared course runtime is implemented.
