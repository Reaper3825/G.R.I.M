# Single-Step Arithmetic Tool v1: semantic role revision

The `semantic_roles_v2` generation revision replaces the 60,000 `ssatv1_` rows
in the existing v1 course. Course and curriculum IDs remain unchanged.
The source contract is [GRIM_REASONING_DATA_CONTRACT.md](GRIM_REASONING_DATA_CONTRACT.md).

## Distribution

There are 12,000 direct arithmetic examples (3,000 per operation) and 48,000
contextual examples. Each contextual relationship has 4,000 cases, with all
three unknown positions represented for every case:

| Relationship | Unknown positions | Rows |
| --- | --- | ---: |
| starting - removed = remaining | starting, removed, remaining | 12,000 |
| starting + added = final | starting, added, final | 12,000 |
| smaller + difference = larger | smaller, difference, larger | 12,000 |
| groups * amount per group = total | groups, amount per group, total | 12,000 |

The resulting operation counts are addition 15,000, subtraction 27,000,
multiplication 7,000, and division 11,000. Equal coverage of unknown roles is
intentional; it does not imply equal operator counts, since inverse additive
relationships require subtraction in two of three positions.

Metric, imperial, and count examples number 20,002, 19,999, and 19,999. The
three-row difference preserves complete contrast triplets. Contexts cover
29 units/items, including length, mass, volume, books, tickets, people,
students, and vehicles. Count quantities and group counts stay integral;
measurements include exact finite decimals. No unit conversions are introduced.

## Semantic variation

The previous generator mapped each operator to a single outcome role, including
subtraction to the remainder. The revised generator chooses a relationship and
the requested unknown first, then selects the operation and operands.

The same underlying case appears with each possible quantity omitted and asked
for. Wording varies across four fact patterns and three Determine formulations,
with both fact orders and question-first/question-last prompts. Comparison
wording includes both "more" and "fewer". Objects and people use appropriate
actions: students join or leave classes, books are lent or received, and water
is used or added. These remain template-generated examples, not a claim of
unrestricted linguistic coverage.

Each contextual triplet shares a `source_sequence_id`. If creating train/eval
splits downstream, keep entire triplets together; for a stronger generalization
test, hold out whole story/wording families as well. The separate curated
six-example evaluation seed remains outside the regenerated training course.
No model evaluation or training is performed by this generator.

The training-time inference diagnostic now cycles through twelve matching
question roles, one probe per configured diagnostic interval. Its first probe
is the original tank-used question. Probes cover liters, feet, marbles, and
students; the `case=` log field identifies the probe without exposing that label
to the model. `GRIM_SAMPLE_PROMPT` overrides the question. Both default and
custom questions use empty knowns/unknowns and the same generic goal as training.
Default limits are 256 new tokens and 2048 displayed characters, overridable
with `GRIM_SAMPLE_TOKENS` and `GRIM_SAMPLE_MAX_CHARS`. These are limits, not a
guarantee of a complete model response. Source-level alignment checks run with
`python -m unittest discover -s scripts -p test_diagnostic_arithmetic_contract.py`;
they do not build or execute the trainer.

## Preserved authoring constraints

- `knowns` and `unknowns` are empty arrays; `update` is empty.
- Determine contains natural-language work and requested roles, without numbers
  or identifiers.
- Define contains exactly two numeric `${variable} -> value;` bindings and one
  `${variable};` declaration.
- Execute contains exactly one `+`, `-`, `*`, or `/` operation on two variables
  inside one `<TOOL>...</TOOL>` span, followed by the result pointer.
- Answer names the requested quantity and references that same result variable;
  numeric tool results are not prefilled.
- Half the tool expressions have parentheses and half do not.
- Every variable name consists of exactly two stems backed by existing
  `stem_` entries in `manual_vocab.json`. Both parts are validated, for example
  `start_volume`, `change_count`, `difference_length`, and `group_count`.
  No vocabulary changes are needed. Measurement units remain explicit in the
  prompt and answer; identifiers use length/weight/volume/count suffixes.
- Goal text is generic and evidence is empty, so the prefix does not reveal the
  selected operation or requested role. Those must be inferred from the prompt.

## Reproduction and checks

```powershell
python -m unittest discover -s scripts -p test_generate_single_step_arithmetic_tool_curriculum.py
python scripts/generate_single_step_arithmetic_tool_curriculum.py --merge-data-dir resources/models/GRIM-text/training/data
```

The generator validates all rows before merging and writes a manifest under
`.codex_tmp/single_step_arithmetic_tool_v1/`. It reports role, operator, category,
wording, and prompt-order counts, manual-vocabulary and JSONL hashes, and the
revision name. The checked-in scenario definitions are in
`scripts/single_step_arithmetic_scenarios.py`.

Tests check the entire 60,000-row contract and balance, both vocabulary parts,
prompt numbers versus supplied operands, integer counts, all three tank
questions, inverse relationships, division output units, malformed candidates,
deterministic seeds, and preservation of unrelated rows during a merge.

For the September 28 regeneration, the original data and registry were backed
up under `.codex_tmp/single_step_arithmetic_tool_v1/backups/20260928_161244/`.
`regeneration_baseline.json` and `regeneration_audit.json` in the output directory
record the before/after verification. The merge changes authored JSONL data;
existing compiled corpora are not regenerated and require the normal data
compilation workflow before a future training run.
