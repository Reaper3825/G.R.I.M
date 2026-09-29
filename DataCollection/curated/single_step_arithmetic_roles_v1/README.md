# Single-step arithmetic: requested quantity supplement

The main 60,000-row generator now incorporates these relationship patterns;
see [the regeneration notes](../../../docs/SINGLE_STEP_ARITHMETIC_TOOL_V1.md).
The files here remain separate review/evaluation seeds and are not separately
merged into the main course.

This curated seed set targets semantic role selection while preserving the v1
Determine / Define / TOOL / Answer format. It contains 30 proposed training
entries (10 contrast triplets) and 6 held-out evaluation entries (2 triplets).
It has not been merged into the live dataset or used for training.

The existing generator maps every subtraction to "amount remaining", every
addition to "combined amount", and every division to "amount per group".
Its contextual subtraction prompts all give the starting and used quantities.
That coverage gap is consistent with the reported inference: the operation and
format can be correct while the selected question and factual bindings are wrong.
It is evidence of a curriculum gap, not proof of the model's internal cause.

## Files

- `train.jsonl`: importable ConceptBlock candidates, including the reported tank prompt.
- `eval.jsonl`: held-out ConceptBlocks; never include these in training.
- `review_manifest.json`: offline role labels and numeric results for review and scoring;
  do not feed this file to the model.
- `scripts/curate_single_step_arithmetic_roles.py` (repository-relative): reproducible authoring source.

## What the contrasts teach

| Relationship | Requested quantity | Required calculation |
| --- | --- | --- |
| initial - removed = remaining | initial | removed + remaining |
| initial - removed = remaining | removed | initial - remaining |
| initial - removed = remaining | remaining | initial - removed |
| initial + added = final | initial | final - added |
| initial + added = final | added | final - initial |
| initial + added = final | final | initial + added |
| groups * amount per group = total | groups | total / amount per group |
| groups * amount per group = total | amount per group | total / groups |
| groups * amount per group = total | total | groups * amount per group |
| smaller + difference = larger | smaller | larger - difference |
| smaller + difference = larger | difference | larger - smaller |
| smaller + difference = larger | larger | smaller + difference |

Each triplet changes which fact is omitted and which quantity the question asks
for. Numbers and story remain related across the triplet. The two tank triplets
also reuse the same quantities across depletion and refill situations. This
breaks an operation-to-answer-role shortcut. "Added" sometimes requires
subtraction; "shorter" sometimes requires addition; division can find either a
group count or an amount per group. Reversed fact order, question-first wording,
integer counts, and decimal measurements provide additional variation.

Determine names the requested outcome and the relationship in ordinary language.
Define binds only the two supplied values and declares the requested unknown.
Execute uses one variable-only tool call and points to that unknown. Answer
references that same variable and states what it means. Knowns, unknowns and
update stay empty. No numeric tool result is authored into the continuation.
Goal fields are deliberately generic: they do not supply the role or operation
that the model should learn to infer from the prompt.

## Use and evaluation

Treat this as a reviewed seed set, not a demonstrated fix or a complete
distribution rebalance. Thirty rows are only 0.05% of the original 60,000-row
generation target. For a larger revision, expand story wording and values across
these unknown positions; duplicating only these rows risks memorization. Choose
the training mix through held-out role-selection results, not syntax accuracy.

Keep entire scenarios together when splitting expanded data. The flour and
auditorium scenarios here are held out in full. Six probes are a smoke test, not
a statistically reliable generalization benchmark. Add independent held-out
increase and comparison scenarios before drawing broad conclusions. The exact
reported tank prompt is a training example here and must not be reported as
held-out success.

Score whether Determine targets the asked-for quantity, whether supplied numbers
are bound to their actual roles, whether the tool expression solves that target,
and whether Answer reports it. Allow different variable names. Score semantic
roles separately from numeric correctness: both 120 - 84 interpretations produce
36, but only one answers how much was used. Evaluate with the same upstream
prefix as deployment; additionally use prompt-only probes to detect dependence
on target-state hints. Do not supply the stored evaluation continuation as input.

Regenerate with `python scripts/curate_single_step_arithmetic_roles.py`.
The existing v1 generator's merge mode replaces the original v1 membership;
do not use it to merge this additive supplement. Integrating these entries requires
appending their IDs to the intended curriculum/course as well as adding the rows.
