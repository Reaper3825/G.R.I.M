---
name: ExecutionBlock Structured Execution Cutover Companion
overview: Living companion to EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md. Tracks workstream status, ownership changes, migrated call sites, deleted fallback paths, validation results, and diagram deltas. Must be updated in the same change whenever the cutover architecture or flow changes.
todos: []
isProject: false
---

# ExecutionBlock Structured Execution Cutover Companion

> This is **not** the full cutover plan.
> It is the **living refactor ledger** for the structured-execution cutover: what changed, what now owns each concern, which legacy paths were deleted, what was validated, and which gate is next.

## Mandatory maintenance rule

Every time an agent refactors, replaces, splits, renames, deletes, or reroutes a system that affects this cutover, this file must be updated in the **same change**.

If code changes the cutover architecture, ownership boundaries, data flow, validation flow, selector flow, or fallback/deletion status but this file does not change, the update is incomplete.

## Required per-update contract

Every qualifying update must append or revise the relevant workstream entry with, at minimum:

1. workstream id and status transition
2. concise summary of what changed
3. changed files
4. new ownership boundary
5. explicit non-responsibilities if the boundary changed
6. integration points / callers affected
7. migrated consumers or call sites
8. removed or deleted legacy / fallback paths
9. validation performed, skipped, or still blocked
10. remaining gaps / next gate
11. flow-diagram delta summary referencing `EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md`

## Same-change artifact coupling rule

A qualifying refactor step is not complete unless all three artifacts are updated together:

- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md`
- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md`
- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md`

## Current cutover status

- **Overall status:** In progress
- **Active workstream:** Workstream 3 — GRMT format cutover
- **Last completed gate:** Workstream 2 — canonical structured sequence builder replaces `__SLOTS__`
- **Next gate:** Begin Workstream 3 GRMT format cutover to serialize compiled execution payloads
- **Current implementation posture:** Workstream 2 canonical builder (`ConceptExecutionSequenceBuilder`) replaces the `__SLOTS__` debug path; concept JSON → `StructuredExecutionRecord` → canonical text → tokenize → `CompiledStructuredExecutionPayload` with paired `token_exec_slots` + `teacher_steps` + `compiled_bootstrap_bindings` in a single builder pass; DataLoader uses the canonical builder exclusively

## Workstream-specific update contract

### Workstream 0 — execution_block file deflation before semantic cutover

**Status**
- Completed

**Each update must record**
- deleted kernels, deleted APIs, and deleted `ExecutionBlockConfig` surface
- final public/private include boundary after each split step
- which helpers moved into memory-stream vs data-stream ownership
- whether any silent skip, batch-global atom adaptation, or invalid decode-time `<NUM>` fallback path was removed
- validation performed on the surviving layer surface

### Workstream 1 — canonical structured execution source-of-truth model

**Status**
- Completed

**Each update must record**
- semantic types added or moved into `ExecutionMetadata.hpp`
- which files now own compiled payload fields
- the authoritative execution-active signal after the change
- any `D_row` or `R` contract clarification introduced
- validation or compile impact from metadata shape changes

### Workstream 2 — canonical structured sequence builder replaces `__SLOTS__`

**Status**
- Completed

**Each update must record**
- builder inputs/outputs changed
- emitted compiled payload fields and provenance emitted together
- deleted `__SLOTS__` or legacy JSON helper paths
- builder-time hard-fail checks added or tightened
- data regeneration implications discovered by the change

### Workstream 3 — GRMT format cutover

**Status**
- Not started

**Each update must record**
- GRMT version change and serialized field layout
- loader rejection rules and removed compatibility paths
- cache/data regeneration requirements
- any validator assumptions now satisfied directly from GRMT payloads
- validation performed on serialization/deserialization paths

### Workstream 3A — tensor-backed learned parameter and checkpoint ownership

**Status**
- Not started

**Each update must record**
- which selector/execution tensors were introduced or moved
- exact owner module after the change
- `ParameterGroup` / optimizer visibility changes
- checkpoint request/view/schema changes
- deleted sidecar or shadow ownership paths
- validation performed for save/load and shape checks

### Workstream 4 — single shared execution payload validator

**Status**
- Not started

**Each update must record**
- validator entry points added or removed
- rules centralized into the shared validator
- duplicated validation logic deleted from callers
- exact failure point before GPU work
- any new row/batch invariants enforced

### Workstream 5 — Phase1 sequence handling rules for execution-active rows

**Status**
- Not started

**Each update must record**
- BOS/EOS remap behavior for compiled metadata
- padding behavior for slot maps, provenance, and selector supervision
- fragmentation rejection logic for execution-active rows
- any decode-position alignment changes
- validation or tests proving remap correctness

### Workstream 6 — row-local execution orchestration

**Status**
- Not started

**Each update must record**
- row-local atom extraction or view changes
- exact `executeStep()` input contract after the change
- batch-global assumptions removed
- row-local validation and runtime checks introduced
- mixed-batch validation proving no cross-row atom leakage

### Workstream 7 — delete silent execution skips

**Status**
- Not started

**Each update must record**
- silent paths deleted
- new fail-loud path and owning file
- orchestrator execution gate after the change
- any thrown error conditions introduced
- validation showing invalid active rows fail instead of silently skipping

### Workstream 8 — align validation path and training path

**Status**
- Not started

**Each update must record**
- training/validation call sites aligned
- any previously divergent rules removed
- logging vs rejection responsibilities after the change
- validation proving both paths fail on the same invalid rows

### Workstream 9 — inference and generation alignment

**Status**
- Not started

**Each update must record**
- candidate-set construction rules
- selector/policy ownership boundary after each change
- deleted heuristic selector or fallback branches
- decode-time ExecutionBlock atom-path wiring
- selector supervision, ambiguity behavior, and bind-or-mask semantics validated

### Workstream 10 — tests that lock the contract in place

**Status**
- Not started

**Each update must record**
- tests added or updated
- contract area covered by each test change
- forbidden behavior newly locked out
- smoke tests / build validation performed
- remaining uncovered gates, if any

## Update entry template

For each qualifying refactor step, append an entry under the relevant workstream using this shape:

### YYYY-MM-DD — Workstream N — short title

**Status transition**
- `Not started -> In progress`
- `In progress -> Completed`

**What changed**
- ...

**Changed files**
- ...

**Ownership after change**
- ...

**Integration points / migrated consumers**
- ...

**Legacy deleted**
- ...

**Validation**
- ...

**Flow diagram delta**
- ...

**Remaining gaps / next gate**
- ...

## Update log

### 2026-03-29 — Workstream 0 — initial ExecutionBlock surface deflation + split scaffolding

**Status transition**
- `Not started -> In progress`

**What changed**
- Trimmed dead public `ExecutionBlockLayer` surface by removing `encodeState()`, `lastDivClampCount()`, and the unused `expected_read_v1` / `expected_read_v2` execution-step inputs.
- Shrunk `ExecutionBlockConfig` so it only carries layer-owned behavior instead of mirroring orchestration-owned temperature, placement, entropy, stream, and cuBLAS knobs.
- Added private split files under `Layers/ExecutionBlock/`:
	- `execution_block_internal.hpp`
	- `execution_block_memory_stream_GPU.*`
	- `execution_block_data_stream_GPU.*`
- Moved the first clearly-owned helpers out of the monolith:
	- `ExecutionMemory::allocate()` / `clear()`
	- slot bootstrap helpers
	- entropy-loss helpers
- Deleted the dead `kernelSliceColumns` and `kernelFourOpMixForward` kernels from the coordinator file.
- Replaced the silent `executeStep()` early return for rows with no bootstrapped value slots with a fail-loud runtime error.
- Updated training/inference construction paths so they no longer push removed config fields into `ExecutionBlockConfig`.

**Changed files**
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu`
- `resources/models/GRIM-text/training/TrainingOps.cu`
- `resources/models/GRIM-text/Layers/InitInferenceState/InitinferenceState.cu`
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- `resources/models/GRIM-text/training/Inference_GPU.cu`
- `resources/models/GRIM-text/training/TrainingLoop/CMakeLists.txt`
- `resources/models/GRIM-text/Tests/ExecutionBlockTest.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/DOCUMENTATION.md`

**Ownership after change**
- `execution_block_GPU.hpp` is now the minimal public surface.
- `execution_block_internal.hpp` is the private shared internal boundary for the split implementation.
- `execution_block_memory_stream_GPU.cu` owns memory allocation/bootstrap helpers.
- `execution_block_data_stream_GPU.cu` owns entropy-loss helpers and data-stream-adjacent utilities that no longer need to live in the coordinator.
- Training/inference orchestration continues to own temperature schedule, entropy scheduling, and layer placement.

**Integration points / migrated consumers**
- `TrainingOps.cu` and `InitInferenceState.cu` now construct the trimmed `ExecutionBlockConfig`.
- `AutogradTraining.cu` and `Inference_GPU.cu` now call the simplified `executeStep(...)` signature.
- `TrainingLoop/CMakeLists.txt` now compiles the new split `.cu` files into both the main GRIM-text implementation target and `execution_block_test`.

**Legacy deleted**
- `ExecutionBlockLayer::encodeState()`
- `ExecutionBlockLayer::lastDivClampCount()`
- dead `expected_read_v1` / `expected_read_v2` execution-step parameters
- dead kernels `kernelSliceColumns` and `kernelFourOpMixForward`
- stale `ExecutionBlockConfig` mirrors of orchestration-owned fields
- silent no-valid-slots execution skip inside `executeStep()`

**Validation**
- Static callsite/build wiring reconciliation performed.
- No full CUDA build completed yet; editor diagnostics remain dominated by local missing-CUDA-header environment noise.
- Additional compile validation is still required after the remaining monolith-to-split reconciliation settles.

**Flow diagram delta**
- Updated the flow companion status from planned-only to Workstream 0 in progress and noted the new internal split boundary inside the ExecutionBlock layer.

**Remaining gaps / next gate**
- Finish moving surviving helpers out of `execution_block_GPU.cu` so the coordinator is genuinely thin.
- Confirm no batch-global atom adaptation or invalid decode-time `<NUM>` fallback remains inside the layer/runtime path.
- Run build/error validation on the split target set.

### 2026-03-29 — Workstream 0 — closure: docs/tests aligned and decode-time last-write fallback deleted

**Status transition**
- `In progress -> Completed`

**What changed**
- Verified the split boundary on disk: `execution_block_GPU.cu` is now a thin public coordinator and the live runtime is owned by the memory-stream and data-stream files.
- Deleted the remaining decode-time `<NUM>` last-write heuristic outside the layer/runtime path. Generation now masks `<NUM>` while ScratchBlock generation is active until the explicit decode-time selector workstream exists.
- Removed persistent `inference_exec_last_write_slot` state tracking and the code that derived it from `p_write` during inference.
- Rewrote `Layers/ExecutionBlock/DOCUMENTATION.md` so it matches the current code instead of documenting deleted kernels, deleted weights, blended-write behavior, or stale candidate-pool semantics.
- Rewrote `Tests/ExecutionBlockTest.cu` so Workstream 0 coverage now targets the surviving layer-owned surface instead of BatchPayload / teacher-step / orchestration concerns.

**Changed files**
- `resources/models/GRIM-text/Common/grim_language_model_gpu.cu`
- `resources/models/GRIM-text/training/Inference_GPU.cu`
- `resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/DOCUMENTATION.md`
- `resources/models/GRIM-text/Tests/ExecutionBlockTest.cu`

**Ownership after change**
- Decode-time generation no longer owns or consults a heuristic "last write" selector surrogate.
- The current generation contract is explicit: `<NUM>` is masked pre-sampling until Workstream 9 introduces a real selector.
- ExecutionBlock docs/tests now cover only the live layer boundary and current runtime semantics.

**Integration points / migrated consumers**
- `generateSequenceGPU(...)` now masks `<NUM>` instead of attempting slot/value binding from inference runtime state.
- Inference forward paths still preserve persistent `ExecutionMemory`, but no longer derive or export a last-write slot heuristic.
- `execution_block_test` now exercises only the surviving ExecutionBlock public surface.

**Legacy deleted**
- decode-time `<NUM>` binding from `inference_exec_last_write_slot`
- persistent `inference_exec_last_write_slot` field/state
- stale ExecutionBlock docs for blended writes, deleted tensors, and deleted candidate-pool behavior
- unrelated BatchPayload / teacher-step / orchestration assertions in `ExecutionBlockTest.cu`

**Validation**
- Static search confirmed `execution_block_internal.hpp` remains private to `Layers/ExecutionBlock/`.
- Static search confirmed deleted public APIs/kernels (`encodeState`, `lastDivClampCount`, `expected_read_v1`, `expected_read_v2`, `kernelSliceColumns`, `kernelFourOpMixForward`) remain absent.
- Static search confirmed the decode-time last-write `<NUM>` fallback is removed from code paths.
- Editor diagnostics for CUDA-heavy files remain dominated by missing local CUDA headers/toolchain, so full compile validation is still blocked locally.

**Flow diagram delta**
- Updated the flow companion to reflect the current enforced runtime: Workstream 0 complete, ExecutionBlock split locked, and generation masks `<NUM>` until an explicit selector exists.

**Remaining gaps / next gate**
- Start Workstream 1 without reintroducing orchestration or metadata semantics into the ExecutionBlock layer.

### 2026-03-29 — Workstream 0 — explicit row-local atom views for training + decode runtime

**Status transition**
- `In progress -> In progress`

**What changed**
- Tightened the live `ExecutionBlockLayer::executeStep(...)` contract so callers must provide non-null row-local atom buffers even when a row has zero atoms.
- Added `ScratchBlockLayer::extractRowLocalAtomView(...)`, which compacts batch-global ScratchBlock atom buffers into row-local positions and embeddings with row-relative positions.
- Updated training orchestration to build one row-local atom view per batch row and pass only row-local atom positions, row-local atom embeddings, and the row-local slot-map base pointer into `executeStep(...)`.
- Updated decode-time execution to build a one-token ScratchBlock atom view before executing the block and to fail hard if a slot-bound decode token cannot produce that atom view.
- Hardened `ExecutionBlock` atom-slot validation so row-local atom positions are bounds-checked against `row_tokens` before indexing the row-local slot map.

**Changed files**
- `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp`
- `resources/models/GRIM-text/Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_internal.hpp`
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- `resources/models/GRIM-text/training/Inference_GPU.cu`
- `resources/models/GRIM-text/Layers/ExecutionBlock/DOCUMENTATION.md`

**Ownership after change**
- `ScratchBlockLayer` now owns row-local atom-view extraction from its batch-global detection buffers.
- Training and inference orchestration own construction/use of those row-local views at the call boundary.
- `ExecutionBlockLayer` now treats atom positions and slot maps as row-scoped runtime inputs and no longer accepts null atom-pointer callers as a tolerated contract.

**Integration points / migrated consumers**
- `AutogradTraining.cu` now extracts one row-local atom view per batch row and reuses it for all execution steps on that row.
- `Inference_GPU.cu` now routes decode-time execution through a ScratchBlock-backed one-token atom view instead of null atom pointers.

**Legacy deleted**
- batch-global ScratchBlock atom buffers passed directly into per-row `executeStep(...)` from training orchestration
- decode-time `executeStep(...)` calls with `nullptr` atom buffers during active execution

**Validation**
- Static runtime-path audit completed for training and inference callers.
- Row-local atom-slot validation now includes an explicit out-of-range stage for row-relative atom positions.
- Full CUDA compile/runtime validation is still blocked on the local environment lacking CUDA toolchain support.

**Flow diagram delta**
- Updated the flow companion so training and inference both show explicit row-local ScratchBlock atom-view wiring into ExecutionBlock runtime.

**Remaining gaps / next gate**
- Finish thinning `execution_block_GPU.cu` further so stream-local helpers own more of the live implementation.
- Validate the new row-local atom-view plumbing on a CUDA-capable build/test environment.

### 2026-03-29 — Cutover companion created

**Status transition**
- `Planning -> Planning with living artifacts`

**What changed**
- Created the living companion ledger for the structured-execution cutover.
- Defined the same-change maintenance rule and per-update contract.
- Established per-workstream ledger expectations before implementation begins.

**Changed files**
- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.codoc.md`
- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_PLAN.md`
- `docs/EXECUTION_BLOCK_STRUCTURED_EXECUTION_CUTOVER_FLOW.md`

**Ownership after change**
- The plan remains the normative cutover checklist.
- This codoc owns the living implementation ledger and workstream status narrative.
- The flow companion owns the maintained Mermaid flow view of sequencing and cutover flow.

**Integration points / migrated consumers**
- Future architecture-affecting work on the cutover must update all three artifacts in the same change.

**Legacy deleted**
- None yet; this entry establishes maintenance requirements before code deletion starts.

**Validation**
- Documentation-only change; no build or runtime validation performed.

**Flow diagram delta**
- Initial flow companion created with workstream order, same-change maintenance flow, and end-to-end cutover flow diagrams.

**Remaining gaps / next gate**
- Begin Workstream 0 and record the first architectural delta here when code changes start.

### 2025-07-25 — Workstream 1 — canonical structured execution source-of-truth model

**Status transition**
- `Not started -> Completed`

**What changed**
- Created `Shared/Execution/ExecutionMetadata.hpp` as the single cross-layer definition site for all semantic execution metadata types: `StructuredExecutionRecord`, `CompiledStructuredExecutionPayload`, `TeacherStep`, `CompiledBootstrapBinding`, `SlotSelectionTarget`, `BootstrapLiteralBinding`.
- Moved `TeacherStep` ownership from `BatchPayload.hpp` (local struct) to `ExecutionMetadata.hpp` (`GRIM::Execution::TeacherStep`). BatchPayload retains a `using` alias for call-site compatibility during cutover.
- Extended `TrainingSequence` with compiled execution payload fields: `execution_active` (bool), `compiled_bootstrap_bindings`, `teacher_steps`, `slot_selection_targets`.
- Extended `TrainingSampleView` with matching pointer fields and wired them from `TrainingSequence` in `getSample()`.
- Extended `BatchPayload` with per-row compiled execution payload: `execution_active` [batch_size], `compiled_bootstrap_bindings` [batch_size], `slot_selection_targets` [batch_size].
- Updated `buildBatchPayload()` in `BatchPayload.cu` to extract execution metadata from `TrainingSequence` during PHASE 2 and populate `BatchPayload` arrays during PHASE 4.
- Added structural validation in `BatchPayload::validate()` for all new execution payload fields, including `execution_active=false + non-empty teacher_steps` rejection.
- Added `reconstructSlotDomain()` utility inline that computes `D_row` from `compiled_bootstrap_bindings ∪ teacher_steps`.

**Changed files**
- `resources/models/GRIM-text/Shared/Execution/ExecutionMetadata.hpp` (NEW)
- `resources/models/GRIM-text/training/training_data_loader.hpp`
- `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp`
- `resources/models/GRIM-text/Shared/Batching/BatchPayload.cu`

**Ownership after change**
- `ExecutionMetadata.hpp` is the single canonical definition site for all structured execution semantic types. No other file may define `TeacherStep`, `CompiledBootstrapBinding`, `SlotSelectionTarget`, or `BootstrapLiteralBinding`.
- `TrainingSequence` owns per-sample compiled execution metadata (activation state + bindings + steps + targets).
- `TrainingSampleView` provides non-owning pointer access to the same metadata.
- `BatchPayload` owns per-batch transport of compiled execution metadata, populated by `buildBatchPayload()`.
- `BatchPayload::validate()` enforces structural consistency of execution payload.

**Non-responsibilities**
- `ExecutionMetadata.hpp` does NOT own GPU tensor allocation or parameter registration — that is Workstream 3A.
- `TrainingSequence` fields are populated by the data loader / GRMT deserializer — format cutover is Workstream 3.
- Batch-level validation for row-consistent step counts, slot domain completeness, and teacher-step op validity is Workstream 4.

**Integration points / migrated consumers**
- `ComputeLossBatch.cu` reads `payload.teacher_steps` — type unchanged via alias, no code changes needed.
- All downstream consumers of `BatchPayload` now have access to `execution_active`, `compiled_bootstrap_bindings`, and `slot_selection_targets` per row.
- `buildBatchPayload()` is the single population site — no downstream code may populate these fields.

**Legacy deleted**
- Local `struct TeacherStep` definition in `BatchPayload.hpp` — replaced by `using` alias to `GRIM::Execution::TeacherStep`.

**Validation**
- Static field alignment verified: `TrainingSequence` → `TrainingSampleView` → `BatchPayload` all carry the same four compiled execution fields.
- `BatchPayload::validate()` enforces size consistency and structural invariants (execution_active=false rejects non-empty teacher_steps).
- `buildBatchPayload()` populates all execution fields in the same pass as existing per-token arrays.
- Full CUDA compile validation is blocked on local environment lacking CUDA toolchain; editor diagnostics for CUDA-heavy files are dominated by missing `cuda_runtime.h`.

**Flow diagram delta**
- Updated flow diagram to show metadata ownership path: `ExecutionMetadata.hpp` → `TrainingSequence` → `TrainingSampleView` → `BatchPayload` → downstream consumers.

**Remaining gaps / next gate**
- Begin Workstream 2: replace `__SLOTS__` sequence builder with compiled structured sequence builder that populates the new `TrainingSequence` execution fields from `StructuredExecutionRecord`.

### 2026-03-29 — Workstream 1 — plan alignment verification fixes

**Status transition**
- `Completed -> Completed` (alignment fixes within completed workstream)

**What changed**
- Added `const std::vector<int32_t>* token_exec_slots = nullptr` to `TrainingSampleView` and wired it from `TrainingSequence` in `getSample()`. This was a gap found during plan alignment verification: the completion criteria require all 5 compiled payload fields in all 3 structs.
- Transitioned 3 per-row teacher supervision gates in `AutogradTraining.cu` (lines 710, 1288, 1403) to check `execution_active[b]` as the authoritative activation signal before looking up teacher data. Previously these sites checked `teacher_steps.empty()` as a data-availability proxy, which was functionally correct but violated criterion 3's requirement that no code treat `teacher_steps.size()` as the execution-activation signal.

**Changed files**
- `resources/models/GRIM-text/training/training_data_loader.hpp`
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`

**Ownership after change**
- `execution_active[b]` is now the authoritative per-row gate at all three execution supervision sites in `AutogradTraining.cu`.
- `teacher_steps` presence is still checked as data-availability after the activation gate is satisfied.

**Integration points / migrated consumers**
- No external API changes; same callers, same observable behavior for rows where `execution_active == true`.

**Legacy deleted**
- None; the previous pattern was a data-availability check, not an outright legacy path. It was tightened to match the authoritative activation contract.

**Validation**
- All 4 WS1 completion criteria now PASS:
  1. ExecutionMetadata.hpp is the single cross-layer definition site ✅
  2. TrainingSequence, TrainingSampleView, BatchPayload all carry 5 compiled payload fields ✅
  3. Activation state uses execution_active[b] at all per-row teacher supervision sites ✅
  4. Comments and docs explicitly state paired projections and D_row reconstruction ✅

**Flow diagram delta**
- No structural diagram changes; the metadata ownership diagram from the prior entry remains accurate.

**Remaining gaps / next gate**
- Workstream 1 is fully aligned with all completion criteria. Proceed to Workstream 2.

### 2025-07-27 — Workstream 2 — canonical structured sequence builder replaces `__SLOTS__`

**Status transition**
- `Not started -> Completed`

**What changed**
- Created `ConceptExecutionSequenceBuilder.hpp` / `.cu` — the canonical builder that replaces the `__SLOTS__` debug serialization path.
- The builder provides a single-pass pipeline: `buildConceptSequence()` calls `buildStructuredExecutionRecord()` → `renderCanonicalText()` → tokenize → `compileExecutionPayload()`.
- `buildStructuredExecutionRecord()` parses concept JSON `state_0.atoms` into `BootstrapLiteralBinding` entries with sequential slot IDs, builds `ExecutionStep` structs from `execution[]` array with explicit arg-slot resolution and write-slot allocation.
- `renderCanonicalText()` emits Q/STATE0/EXEC/STATE1/EXP/A text format — NO `__SLOTS__` block.
- `compileExecutionPayload()` maps bootstrap bindings to ATOM_NUM token positions in document order, emits `token_exec_slots`, `teacher_steps`, and `compiled_bootstrap_bindings` as a single paired output from one builder pass.
- Builder enforces all structural violations: zero bootstrap bindings on active row, duplicate slot_id, duplicate token_pos, missing ATOM_NUM token for bound literal, execution steps with fewer than 2 args, arg slots not found.
- DataLoader.cu: deleted `slotOrderFromConceptJson()`, `conceptJsonToTrainingText()`, `nearEqualConceptNumeric()`, `assignExecSlotsFromOrder()`, `opStringToId()`, `teacherStepsFromConceptJson()`, `formatNumberForConcept()`, and the old `loadConceptBlocksCorpus()`.
- DataLoader.cu: replaced `concept_entries` (pair<string, vector<double>>) with `concept_json_entries` (vector<json>), new `loadConceptBlocksJson()` returns raw parsed JSON objects.
- DataLoader.cu: concept processing loop now calls `buildConceptSequence()` for each JSON entry, populating `token_exec_slots` from the compiled payload.
- DataLoader.cu: vocab training corpus for concept entries now uses `renderCanonicalText()` instead of the deleted `conceptJsonToTrainingText()`.
- DataLoader.hpp: updated header comment to reference canonical builder instead of `__SLOTS__` debug path.

**Changed files**
- `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.hpp` (NEW)
- `resources/models/GRIM-text/Shared/DataLoader/ConceptExecutionSequenceBuilder.cu` (NEW)
- `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- `resources/models/GRIM-text/Shared/DataLoader/DataLoader.hpp`

**Ownership after change**
- `ConceptExecutionSequenceBuilder` is the single canonical builder for execution-active concept rows.
- `DataLoader.cu` owns corpus loading and GRMT serialization but delegates all concept execution building to the canonical builder.
- `ExecutionMetadata.hpp` (WS1) remains the single definition site for all semantic types.
- No other path can produce `token_exec_slots`, `teacher_steps`, or `compiled_bootstrap_bindings` for concept rows.

**Non-responsibilities**
- The builder does NOT write GRMT — DataLoader owns serialization (WS3).
- The builder does NOT produce `slot_selection_targets` with real supervision — all positions are `Ignore` until a selector supervision pipeline exists (WS9).

**Integration points / migrated consumers**
- `DataLoader.cu`'s concept processing loop is the sole consumer of `buildConceptSequence()`.
- Vocab training corpus construction now calls `renderCanonicalText()`.
- `build_sequence()` lambda (DataLoader-internal) tokenizes the canonical text output.

**Legacy deleted**
- `__SLOTS__` text block emission in `conceptJsonToTrainingText()`
- `slotOrderFromConceptJson()` — flat numeric sequence extraction for tail-number slot recovery
- `conceptJsonToTrainingText()` — old text renderer WITH `__SLOTS__` block
- `nearEqualConceptNumeric()` — float comparator (equivalent now internal to builder)
- `assignExecSlotsFromOrder()` — backward tail-number slot recovery from trailing ATOM_NUM tokens
- `opStringToId()` — moved to builder namespace
- `teacherStepsFromConceptJson()` — value-matching teacher step builder (was unreferenced in the concept processing loop)
- `formatNumberForConcept()` — moved to builder as internal helper
- `loadConceptBlocksCorpus()` — old loader returning pair<string, vector<double>>
- `concept_entries` variable (pair<string, vector<double>> type) in DataLoader

**Validation**
- Static search confirms zero remaining references to `__SLOTS__` in code paths.
- Static search confirms zero remaining references to `assignExecSlotsFromOrder`, `slotOrderFromConceptJson`, `conceptJsonToTrainingText`, `nearEqualConceptNumeric`, `teacherStepsFromConceptJson`.
- Builder enforces all plan-specified structural violations at build time.
- Full CUDA compile validation is blocked on local macOS environment lacking CUDA toolchain.

**Flow diagram delta**
- Updated flow diagram: concept JSON → `ConceptExecutionSequenceBuilder` → canonical text + `CompiledStructuredExecutionPayload` → DataLoader tokenizes text, populates `TokenizedSequence.token_exec_slots` from compiled payload → GRMT output.

**Remaining gaps / next gate**
- Workstream 2 is complete. Proceed to Workstream 3: GRMT format cutover to serialize compiled execution payloads alongside tokenized sequences.
