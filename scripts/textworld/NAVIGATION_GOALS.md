# Live navigation Goal generation

Open World Lab's Live view to refresh `resources/models/games/live_navigation.json`,
then run from the repository root:

```powershell
python scripts/textworld/generate_navigation_course.py
```

Optional inputs:

```powershell
python scripts/textworld/generate_navigation_course.py --snapshot resources/models/games/live_navigation.json --output-dir DataCollection/curated/live_navigation_goals_v1
```

The generator creates one arrival assignment per provider room, starting at
System Hub. The hub itself is excluded because arrival there is initially satisfied.
Goal text uses the snapshot's display name, for example "go to the Voice room."
Snapshot paths and provider identities are retained only in the authoring manifest;
the generator does not change or reconstruct the exported paths.
Provider additions/removals are reflected by rerunning against a fresh snapshot.
No hardcoded room list or third-party Python dependencies are needed.

Outputs are `goals.jsonl` (Goal-only ConceptBlocks), `course.json` (one Navigate
course record, not a complete registry), and `manifest.json` (private authoring
bindings and source snapshot provenance). Default output is
`DataCollection/curated/live_navigation_goals_v1`. Regeneration replaces these files.
The active dataset/registry is not modified.

Assignment IDs depend on provider type and exact registration name. Restarts,
provider generation replacement, room numbering, display names, and path changes
do not change IDs. Display-name changes refresh Goal text; path changes refresh
manifest metadata. Provider renames
create new assignments. Bind destinations using the manifest's provider identity;
its room IDs, paths, and instance values describe only the source snapshot.

LiveWorld has no implemented Goal binding/evaluation lifecycle. Criteria therefore
omit optional evidence and remain unverified. The manifest does not become model
input. The course environment ID is empty because a live session ID is not a
registered TextWorld environment ID. Do not schedule this course through the
TextWorld environment loader. A future live capture/controller must resolve the
provider against its current topology and implement verification explicitly.

These files contain no authored reasoning, Execute commands, or final Answers,
and are not completed SFT demonstrations. No capture or runtime execution occurs.

```powershell
python -m unittest discover -s tests -p test_generate_navigation_course.py
```
