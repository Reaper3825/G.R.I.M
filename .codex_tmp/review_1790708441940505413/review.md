# Review: training_1790708441940505413

## Primary finding: the compiled corpus contains the old curriculum

The regenerated role-based JSONL reached Bridges-2, but the FlatBuffer and compiled arithmetic GRMT still contain old v1 examples. This run therefore does not establish whether the new curriculum teaches the requested role relationships.

A read-only remote audit found:

| Artifact | Evidence |
|---|---|
| concept_blocks.jsonl | Four sampled tail rows have `synthetic_single_step_arithmetic_tool_v1_roles` provenance, varied unknown roles, and the new variable stems. |
| concept_blocks.fb | All 24 sampled arithmetic rows have old provenance and the old four Determine templates. |
| single_step_arithmetic_tool_v1.grmt | Header contains 60,000 sequences. The first decoded sequence is the old `ssatv1_000000`, including `start_millimeters`, the old Determine, and an input goal explicitly naming `total_millimeters`. |

These are bounded content samples, not an exhaustive remote corpus audit. They directly demonstrate stale compiled content. Evidence is saved in [remote_input_audit.json](remote_input_audit.json).

The old goal also exposes the target variable and unit, with operation-specific evidence. The diagnostic now uses the generic goal and empty evidence. Consequently, tiny validation loss on the old corpus is compatible with poor generation under the new diagnostic prefix: training supplies cues that inference does not.

## Why the stale content survived

`ConceptBlockCorpusReader.cu:112` prefers the FlatBuffer, rejecting it only when JSONL has a later modification time. On Bridges-2, the JSONL timestamp was September 29 at 14:18:02 EDT and the FlatBuffer timestamp was 14:18:53. The old FlatBuffer therefore appears newer after transfer. The GRMT timestamp was 14:24:02, before this run started at 15:00:41.

Separately, `ConceptBlockGrmtCompiler.cu:70` reuses an existing GRMT when its tokenizer bundle validates, without checking the source content. Refreshing JSONL alone does not invalidate either downstream artifact. `DataCollection/dataset_target.cpp:633` also uses modification times when deciding whether to migrate JSONL to FlatBuffer.

The earlier regeneration updated JSONL without refreshing FlatBuffer. That remaining step should have been called out explicitly.

## What the outputs actually do

There are 306 sampling attempts: 288 returned outputs and 18 failures reporting nested or mismatched local-atom boundaries.

In the last 60 returned outputs:

| Check | Result |
|---|---:|
| Correct requested operator and operand pair | 13 / 60 |
| All four closing section tags present | 55 / 60 |
| Parsed addition expressions | 43 |
| Parsed subtraction expressions | 6 |
| Parsed multiplication expressions | 1 |
| Parsed division expressions | 0 |
| Unparseable tool expressions | 10 |

The expression check resolves variable bindings and allows commuted operands for addition and multiplication. It does not establish full answer correctness, unit correctness, consistent variable references, or successful tool execution. These are repeated diagnostic cases, not an independent held-out benchmark.

Examples from the latest output for each case:

- Tank question asking how much was used: generates 120 + 84, with ounces/tool wording, instead of 120 - 84 (log line 207721).
- Equal groups question asking amount per group: generates 72 + 8 instead of 72 / 8 (line 206302).
- Question asking the original amount: correctly selects 36 + 84, but mixes students and milligrams (line 216915).
- Comparison difference: generates 50 + 32 and inconsistent result references instead of 50 - 32 (line 212691).

Formatting is more reliable than semantic selection, but formatting and reference consistency are not fully solved either. Extracted samples and line numbers are in [samples.jsonl](samples.jsonl); aggregate statistics are in [summary.json](summary.json).

## Graph interpretation

Reviewed the 14 main diagnostic PNGs and summarized the raw CSV streams. The individual atlas panels were not all visually inspected.

- **Loss:** falls from 3.447 to below 0.001 after approximately 131 optimizer updates in this stage. Both logged validations round to 0.0001. The final recorded training loss is approximately 0.00000287. This demonstrates easy prediction of these training targets, not mastery of the question's requested unknown.
- **Gradient and learning rate:** gradients become very small as token loss vanishes. The 0.0006 learning-rate plateau matches the configured schedule: decay begins halfway through 16,200 planned optimizer steps, while this capture covers approximately 6,125. A flat learning-rate graph here is not evidence of a broken scheduler.
- **Step interpretation:** telemetry global steps 25,363–37,611 include inherited pretraining counters. This is roughly 2.27 arithmetic epochs in the captured run, not 37,000 arithmetic optimizer updates. The capture ends mid-run.
- **Representation and RMS plots:** final rho stays approximately 0.106–0.301, with median 0.184; it does not show complete directional collapse. RMS spread rises from approximately 1.82 to 17.11, and activation growth is substantial. These deserve monitoring on the corrected corpus, but do not identify the cause of semantic failure. RMS gamma values were already below one at this stage's start; their entire distance from one cannot be attributed to this run.
- **Adam plots:** the reported cumulative learning-rate/Xavier-scale ratio is a proxy, not a measurement of parameter displacement. Its label does not establish destructive weight updates. Likewise, the plotted signal-dominance formula is not measured task learning.
- **Execution plots:** zero execution targets/loss do not certify successful arithmetic tool use. This corpus supervises authored TOOL text; these execution telemetry streams do not score the generated expressions against the question.
- **Other flat streams:** all-zero hardware-alignment or unigram diagnostic streams provide no affirmative health evidence without confirming instrumentation is active. Static positional-bias parameters are expected to remain static.

## Recommended order of work

1. **Repair and verify the artifact chain before another training experiment.** Refresh FlatBuffer from the new JSONL, then rebuild only the arithmetic GRMT against the existing tokenizer. Preserve `vocab.bin` and token IDs for checkpoint compatibility; do not use a vocabulary rebuild as a cache-invalidation workaround. Verify decoded compiled rows across every role, generic goals, empty knowns/unknowns, permitted variable stems, and single-operation expressions.
2. **Make stale-source detection reliable.** Record source-content and tokenizer hashes with the compiled corpus. Invalidate on content changes rather than transfer-dependent modification times or mere file existence. Include role counts and representative decoded rows in the preflight output.
3. **Run the corrected arithmetic stage from the intended pretraining checkpoint.** Treat that as a new curriculum experiment, with explicit optimizer/scheduler policy, rather than extending the current old-curriculum SFT state and confusing the comparison.
4. **Score meaning separately from token loss.** For each generated response, check requested unknown, operator, operand order, result variable, unit, and answer-reference consistency. Count malformed generations as failures. Include matched stories asking for different unknowns, paraphrases, and held-out templates/scenario groups so near-duplicate role variants cannot make validation misleading.
5. **Only then tune training.** If generation still fails on verified new inputs, investigate shortcut learning, teacher-forced versus free-generation behavior, and activation imbalance with controlled experiments. These logs do not justify choosing a new learning rate or architecture yet.

No trainer, model server, or runtime target was built or run for this review. Remote inspection was read-only; no training job was stopped or restarted.
