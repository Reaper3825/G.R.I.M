# GRIM internal deterministic state episodes v1

Nine nontrivial Goal assignments for the unchanged `grim_internal.grimworld.json`.
Every episode begins at its authored System Hub spawn with the catalog closed.
The generator exhaustively searches the ten reachable symbolic states under
adjacent navigation and catalog open/close actions, excluding the initial state.
It uses actual worker transitions and deduplicates authoritative fact sets.

| Goal family | Episodes | Commands per witness |
| --- | ---: | ---: |
| Reach perception/tools/computer/memory, leave catalog closed throughout | 4 | 1 |
| Open catalog, finish in tools | 1 | 2 |
| Open catalog, finish in System Hub | 1 | 3 |
| Open catalog, finish in perception/computer/memory | 3 | 4 |

Each Goal requires both `player_at(room)` and `open(tool_catalog)` or
`closed(tool_catalog)` simultaneously. Closed-catalog tasks also forbid
`open(tool_catalog)` as a constraint violation. All requirements have structured
evidence; evidence and saved command sequences are private verifier/author data.
Model-facing location goals and criteria address GRIM as "you" (for example,
"You are in the computer room."); constraints use "Do not...". Internal
evidence identifiers such as `player_at` retain their native spelling.
Prompts request actions (for example, "Open the tool catalog in the tools room,
then return to the System Hub. Leave the catalog open."). They do not assert
that the desired terminal state already holds; fresh worker observations supply
the actual starting state.
No Goal claims that tools were inspected or invoked, information was stored, or
that a room visit performed a subsystem operation.

## Files and integration

- `goals.jsonl`: ConceptBlock Goal assignments, with no authored continuations.
  These are not SFT demonstrations. They can be imported through the existing
  ConceptBlock JSONL path. Assign their course to environment
  `env-4056413a539c4244b300030bbd119462` for scheduled environment interaction.
- `manifest.json`: ordered source IDs, witness filenames, command counts,
  authoritative final requirement results, and environment/evaluator identities.
- Repository-root `resources/models/games/episodes/*.grimepisode.json`: the nine witness filenames listed in
  the manifest use the existing version-1 World Lab episode format and filename
  hash. Each embeds the frozen world, Goal, commands, and verified witness identity.

The generator checks live execution, then two isolated replays (including saved
witness identity validation). BFS finds shortest paths for this action subset.
The native catalog provider is declared, but no live registry is connected or
invented; these files contain no generated observation transcript. Catalog state
is verifiable independently of registry contents. Recurrent SFT serialization,
model-generated Answer submission, and training are not implemented by this set.
The active database and curriculum registry are not modified automatically.

Complete goal-following authoring examples are now available in
[demonstrations/examples.md](demonstrations/examples.md), with ordered JSONL
events, separate private verification receipts, and an explicit catalog fixture.
See [demonstrations/README.md](demonstrations/README.md) for their boundaries and
regeneration instructions. Native recurrent training ingestion remains pending.

## Regeneration and checks

From PowerShell at `D:\G.R.I.M`, using the already-installed image:

```powershell
docker run --rm --network none --mount 'type=bind,source=D:\G.R.I.M,target=/workspace' --workdir /workspace/scripts/textworld grim-textworld:1.7.0 python generate_internal_episodes.py --root /workspace
docker run --rm --network none --mount 'type=bind,source=D:\G.R.I.M,target=/workspace,readonly' --workdir /workspace grim-textworld:1.7.0 python -B tests/test_internal_episodes.py
```

No packages are installed and no GRIM executable, model, or training target runs.
Search budget exhaustion fails before publishing results. Regeneration writes
deterministic JSON without timestamps. Old witness identities are rejected after
world/Goal/evaluator changes; old files are not deleted automatically.

Tests cover reproducibility, all nine distinct goals, shortest command counts,
incomplete-prefix rejection, forbidden-opening failure, stale-witness rejection,
and search-budget exhaustion. This small set is a bootstrap curriculum; its
fixed topology does not establish generalization to other worlds or live tools.
