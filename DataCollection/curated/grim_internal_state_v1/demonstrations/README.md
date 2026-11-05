# Goal-following demonstrations

Start with [examples.md](examples.md) to review all nine complete attempts.
Each starts from a fresh System Hub observation and contains a short Determine,
one Execute per native command, its actual permitted feedback, and a final Answer.
There are 21 commands in total. No Define/Update phases are manufactured.

- `examples.jsonl` stores ordered authoring events. `input` is masked context;
  assistant events have `supervised: true`, environment events have `false`.
  Span names are logical names, not hardcoded tokenizer delimiter IDs.
- `verification.private.jsonl` contains evidence, native and fixture-resolved
  receipts, witness identities, and final verification. Never include this file
  in model input or supervised targets. "Private" identifies a data boundary,
  not a filesystem access control.
- `host_fixture.json` explicitly supplies an empty model-tool registry. Native
  catalog opening produces a provider request; its feedback is resolved using
  the empty-registry branch of `ui/world_lab/world_observation.hpp`. This is a
  fixture response, not a recording of the currently running host registry.
- `manifest.json` records example identities, counts, and integration status.

Navigation/container state and all success checks come from real TextWorld
worker replays. The fixture changes only host catalog feedback, not world facts.
Unsupported provider requests fail instead of inventing results. Initial input
and each feedback event use only feedback, description, inventory, location,
and location path; evidence, facts, admissible commands, and outcomes stay out.
Closed-catalog answers say that no opening was performed rather than claiming
to have observed a remote closed container. Open-catalog answers use prior
opening feedback, unchanged intervening actions, and the final observed location.

Determine/Answer text is authored by the generator; these are teacher
demonstrations, not model rollouts. Each Answer is written only after full
evidence success, but no native model-controller Answer grading run is claimed.

## Regenerate and verify

From PowerShell in `D:\G.R.I.M`, using the existing image:

```powershell
docker run --rm --network none --mount 'type=bind,source=D:\G.R.I.M,target=/workspace' --workdir /workspace/scripts/textworld grim-textworld:1.7.0 python -B generate_internal_demonstrations.py --root /workspace
docker run --rm --network none --mount 'type=bind,source=D:\G.R.I.M,target=/workspace,readonly' --workdir /workspace grim-textworld:1.7.0 python -B tests/test_internal_demonstrations.py
```

All witness validation and replay completes before output is written. If goals
or witnesses change, first rerun `generate_internal_episodes.py`.

These files create the requested examples and preserve action/feedback and
supervision boundaries. They are not yet accepted by the native ConceptBlock
compiler or trainer. Native recurrent ingestion, configured span encoding, and
token-level supervision projection remain pending. Do not flatten these events
into one scalar Execute or import the private verification rows as ConceptBlocks.
