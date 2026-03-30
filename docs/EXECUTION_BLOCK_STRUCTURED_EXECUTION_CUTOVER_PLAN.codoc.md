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
- **Active workstream:** Workstream 8 — align validation path and training path
- **Last completed gate:** Workstream 7 — delete silent execution skips
- **Next gate:** Begin Workstream 8 alignment of training and validation paths
- **Current implementation posture:** Workstream 7 completed: orchestrator gates all `executeStep()`, `bootstrapMemoryFromSlotMap()`, `crossAttentionRead()`, and trace-state allocation on `execution_active[b]` from compiled payload activation state; non-execution rows skip before any GPU work; execution-active rows with missing bootstrap data throw immediately; silent bootstrap skip in Inference_GPU.cu replaced with hard throw; no silent early-return path remains anywhere in the execution runtime

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
- Completed

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
- Complete

**Update log**

1. **Validator created** — `ExecutionPayloadValidation.hpp` (header) and `ExecutionPayloadValidation.cu` (implementation, ~280 lines) in `Shared/Execution/`.
2. **Validator entry points:**
   - `buildBatchPayload()` — called after `payload.validate()`, before return. Signature extended with `execution_num_slots`, `execution_num_ops`, `execution_num_steps` params; all 3 Phase2 callers updated.
   - `computeLossBatch()` — called after `payload.validate()`, before any GPU work. Uses `cfg.execution_block_num_slots/ops/steps`.
   - `autogradTrainingStep()` — called after `payload.validate()`, before any GPU work. Uses `model.getConfig()` for params.
3. **Rules centralized into shared validator:**
   - Batch-level: execution_active / compiled_bootstrap_bindings / teacher_steps / slot_selection_targets dimension checks against batch_size
   - Non-execution row: teacher_steps empty, bindings empty, all slots == -1
   - Execution-active row: non-empty bindings, exactly num_steps teacher steps, bootstrap binding injectivity (token_pos + slot_id), teacher step slot/op range checks, D_row reconstruction via `reconstructSlotDomain()`, R↔compiled_bootstrap_bindings mutual consistency, selector supervision target validation (Slot/Null/Ignore), dense selector supervision length == seq_len
4. **Duplicated validation deleted from callers:**
   - Deleted from `ComputeLossBatch.cu`: inline `<NUM>` token slot range check loop (~10 lines) and teacher-steps size + per-field range validation loop (~30 lines)
5. **CMake:** Added `ExecutionPayloadValidation.cu` to both `grim_language_model_gpu_impl` and `execution_block_test` targets
6. **No new legacy paths.** No backwards compatibility code added.

**Changed files**
- `Shared/Execution/ExecutionPayloadValidation.hpp` — NEW
- `Shared/Execution/ExecutionPayloadValidation.cu` — NEW
- `Shared/Batching/BatchPayload.hpp` — signature extended with execution config params
- `Shared/Batching/BatchPayload.cu` — include + validator call added, signature extended
- `Shared/Loss/ComputeLoss/ComputeLossBatch.cu` — include + validator call added, duplicate inline validation deleted
- `training/Autograd/AutogradTraining.cu` — include + validator call added
- `training/Phases/Phase2_TrainingLoop.cu` — all 3 buildBatchPayload callers updated with execution config params
- `training/TrainingLoop/CMakeLists.txt` — new .cu added to both targets

### Workstream 5 — Phase1 sequence handling rules for execution-active rows

**Status**
- Complete

**Each update must record**
- BOS/EOS remap behavior for compiled metadata
- padding behavior for slot maps, provenance, and selector supervision
- fragmentation rejection logic for execution-active rows
- any decode-position alignment changes
- validation or tests proving remap correctness

**Summary of changes**

File modified: `training/Phases/Phase1_Startup.cu`

- **BOS insertion remap**: When BOS is inserted at position 0, all `compiled_bootstrap_bindings[].token_pos` are incremented by 1 to maintain position-sensitive alignment. A `SlotSelectionTarget{Ignore, -1}` is inserted at position 0 of `slot_selection_targets`.
- **EOS insertion remap**: When EOS is appended at the tail, a `SlotSelectionTarget{Ignore, -1}` is appended to `slot_selection_targets`. No binding remap is needed because EOS at the tail does not shift existing positions.
- **Padding preservation**: `PadToSeqMaxLen` resizes `slot_selection_targets` to `max_seq_len` filled with `SlotSelectionTarget{Ignore, -1}`. `compiled_bootstrap_bindings`, `teacher_steps`, and `execution_active` are not position-indexed arrays and are preserved without modification.
- **Sliding-window rejection**: Execution-active rows that exceed `max_seq_len` throw `std::runtime_error` immediately instead of being fragmented. Non-execution rows continue on the existing sliding-window path.
- **Short sequence copy**: The default copy constructor preserves all execution fields (`execution_active`, `compiled_bootstrap_bindings`, `teacher_steps`, `slot_selection_targets`, `token_exec_slots`) for sequences that fit within `max_seq_len`.

### Workstream 6 — row-local execution orchestration

**Status**
- Completed

**Summary**
- Removed dead `atom_embeddings` parameter from entire `executeStep()` chain — was void-cast in coordinator, never consumed by data-stream path
- Renamed `total_tokens` → `row_tokens` in `bootstrapMemoryFromSlotMap()` — callers always passed row-local `sl`, parameter name was misleading
- Removed unused `total_tokens` from internal `crossAttentionReadImpl()` — function body never referenced it
- Verified all internal kernels (`kernelValidateAtomSlots`, `kernelReduceMeanForward`, `kernelBootstrapSlotValues`, cross-attention) operate on row-local bounds
- H tensor remains batch-global `[total_tokens, d_model]` in `executeStep()` — this is correct because backward GradFns (ReduceMeanGradFn, ExecutionBlockInjectGradFn) need full H shape for gradient scattering

**Changed files**
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp` — removed `atom_embeddings` from `executeStep()` and `validateExecuteStepInputsOrThrow()` declarations; renamed `total_tokens` → `row_tokens` in `bootstrapMemoryFromSlotMap()` declaration
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu` — removed `atom_embeddings` from `executeStep()` wrapper and `validateExecuteStepInputsOrThrow()` impl; removed atom_embeddings null check; updated `crossAttentionReadImpl()` call to drop `total_tokens`
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.hpp` — removed `atom_embeddings` from `executeStepCoordinatorImpl()` declaration
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu` — removed `atom_embeddings` from `executeStepCoordinatorImpl()` definition; deleted `(void)atom_embeddings;` line
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.hpp` — removed `total_tokens` from `crossAttentionReadImpl()` declaration
- `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu` — renamed `total_tokens` → `row_tokens` in `bootstrapMemoryFromSlotMap()` impl; removed `total_tokens` from `crossAttentionReadImpl()` definition
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` — removed `row_atom_view.atom_embeddings.data` argument from `executeStep()` call
- `resources/models/GRIM-text/training/Inference_GPU.cu` — removed `row_atom_view.atom_embeddings.data` argument from `executeStep()` call
- `resources/models/GRIM-text/Layers/ExecutionBlock/DOCUMENTATION.md` — updated to reflect deleted `atom_embeddings` from contract

**New ownership boundary**
- `executeStep()` accepts only: `atom_positions` (row-local), `token_to_slot_map` (row-local), `num_atoms` (row-local count), `total_tokens` + `token_offset` + `row_tokens` (batch-global H windowing)
- `bootstrapMemoryFromSlotMap()` accepts `row_tokens` (row-local count, renamed from misleading `total_tokens`)
- `crossAttentionReadImpl()` accepts only `token_offset` + `row_tokens` (no `total_tokens` — unused)

**Explicit non-responsibilities**
- `atom_embeddings` no longer part of `executeStep()` contract — consumed only by ReasoningHead and loss paths
- `extractRowLocalAtomView()` still produces `.atom_embeddings` in `RowLocalAtomView` struct for non-ExecutionBlock consumers

**Deleted legacy / fallback paths**
- Dead `atom_embeddings` parameter thread through 6 function signatures
- `(void)atom_embeddings;` void-cast in coordinator
- `atom_embeddings != nullptr` validation check in `validateExecuteStepInputsOrThrow`

**Validation performed**
- Verified all 3 call sites (`AutogradTraining.cu`, `Inference_GPU.cu`, `execution_block_GPU.cu` thin wrapper) updated consistently
- Verified all 4 signature sites (`.hpp` declaration, `.cu` impl, data_stream `.hpp`, data_stream `.cu`) match
- Verified `crossAttentionReadImpl` body truly does not reference `total_tokens`
- Verified `bootstrapMemoryFromSlotMap` callers (`AutogradTraining.cu`: passes `sl`, `Inference_GPU.cu`: passes `1`) are already row-local
- Verified `kernelValidateAtomSlots` bounds-checks against row-local `num_atoms` and `row_tokens`
- CUDA build validation deferred (no local CUDA toolchain)

**Remaining gaps / next gate**
- WS7: delete silent execution skips — completed: orchestrator gates execution on compiled payload activation state; fail-loud for execution-active rows with missing bootstrap

### Workstream 7 — delete silent execution skips

**Status**
- Completed

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

### 2026-03-29 — Workstream 3 — GRMT v11 compiled structured-execution payload serialization

**Status transition**
- `Not started -> Completed`

**What changed**
- Bumped `GRMT_FORMAT_VERSION` from `10` to `11` in `grim_model_serialization_version.hpp` with full version comment documenting the new binary layout.
- Extended `TokenizedSequence` (DataLoader.cu internal struct) with four new compiled execution payload fields: `execution_active`, `compiled_bootstrap_bindings`, `teacher_steps`, `slot_selection_targets`.
- Updated concept processing loop in DataLoader.cu to transfer all five compiled payload fields from `ConceptExecutionSequenceBuilder` output into `TokenizedSequence` (previously only `token_exec_slots` was transferred).
- Updated `save_grmt` lambda to serialize the full compiled structured-execution payload after `token_exec_slots[len]`:
  - `uint8_t execution_payload_active`
  - `uint32_t compiled_bootstrap_binding_count` + bulk `CompiledBootstrapBinding[count]` (12 bytes each, `static_assert`-guarded)
  - `uint32_t teacher_step_count` + bulk `TeacherStep[count]` (20 bytes each, `static_assert`-guarded)
  - `uint32_t slot_selection_target_count` + per-element field-by-field `SlotSelectionTarget` (uint8 kind + int32 slot_id, 5 bytes each to avoid struct padding)
- Updated `loadGRMTFormat()` in `training_data_loader.hpp` to deserialize all v11 payload fields directly into `TrainingSequence`:
  - `execution_active` from `uint8_t`
  - `compiled_bootstrap_bindings` bulk-read with `static_assert` size guard
  - `teacher_steps` bulk-read with `static_assert` size guard
  - `slot_selection_targets` read field-by-field (kind as `uint8_t`, slot_id as `int32_t`)
- Removed conditional `version >= 10` gate on `token_exec_slots` read — v11 always includes it.
- GRMT v10 rejection is implicit: existing `version != GRMT_FORMAT_VERSION` check rejects any non-v11 file with a clear fatal error message instructing the user to delete and regenerate `.grmt` files.
- Auto-rebuild: `PrepareTrainingDataFromCache()` already compares GRMT header version against `GRMT_FORMAT_VERSION`; old v10 files trigger `grmt_version_mismatch = true` and automatic regeneration.

**Changed files**
- `resources/models/GRIM-text/Common/grim_model_serialization_version.hpp`
- `resources/models/GRIM-text/Shared/DataLoader/DataLoader.cu`
- `resources/models/GRIM-text/training/training_data_loader.hpp`

**Ownership after change**
- `grim_model_serialization_version.hpp` is the single source of truth for `GRMT_FORMAT_VERSION`.
- `save_grmt` in DataLoader.cu owns the GRMT write path and serializes all compiled execution payload fields.
- `loadGRMTFormat()` in `training_data_loader.hpp` owns the GRMT read path and deserializes compiled payload directly into `TrainingSequence`.
- Loader does NOT fabricate or reconstruct execution metadata — all fields come from the serialized GRMT stream.

**Non-responsibilities**
- `save_grmt` does NOT store learned execution/selector weights, optimizer state, or tensor layout — that is Workstream 3A (checkpoint serialization).
- Loader does NOT infer execution-active status from `teacher_steps.size()` — it reads the explicit `execution_payload_active` flag.
- No `D_row` serialization — runtime reconstructs it from `compiled_bootstrap_bindings` ∪ `teacher_steps` per the plan contract.

**Integration points / migrated consumers**
- `TrainingSequence` now has all compiled execution payload fields populated directly from GRMT load, removing the previous gap where only `token_exec_slots` survived serialization.
- `TrainingSampleView` (via `getSample()`) and `BatchPayload` (via `buildBatchPayload()`) automatically see the populated fields — no additional wiring changes needed for downstream consumers.
- `ConceptExecutionSequenceBuilder` output flows through `TokenizedSequence` → `save_grmt` → GRMT file → `loadGRMTFormat()` → `TrainingSequence` with full metadata preservation.

**Legacy deleted**
- Conditional `version >= 10` gate on `token_exec_slots` read — v11 always includes this field.
- GRMT v10 files are no longer loadable (no dual loader, no translation shim, no compatibility mode).

**Validation**
- Serialization format uses `static_assert` to verify `CompiledBootstrapBinding` is exactly 12 bytes and `TeacherStep` is exactly 20 bytes for safe bulk binary read/write.
- `SlotSelectionTarget` is serialized field-by-field (uint8 + int32) to avoid struct padding ambiguity.
- Version rejection: loader's existing `version != GRMT_FORMAT_VERSION` check rejects non-v11 files with explicit regeneration instructions.
- Auto-rebuild: DataLoader's `PrepareTrainingDataFromCache()` detects version mismatch in GRMT header and forces full regeneration.
- Full CUDA compile validation is blocked on local macOS environment lacking CUDA toolchain.

**Flow diagram delta**
- Updated flow diagram to show GRMT v11 serialization carrying full compiled execution payload alongside token data; updated workstream status and structured execution flow status.

**Remaining gaps / next gate**
- Workstream 3 is complete. Proceed to Workstream 5: Phase1 BOS/EOS/padding remap semantics for execution-active rows.

### 2026-03-30 — Workstream 7 — delete silent execution skips

**Status transition**
- `Not started -> Completed`

**What changed**
- Added per-row `execution_active[b]` gating in `AutogradTraining.cu` row loop: non-execution rows `continue` before `ExecutionMemory::allocate()`, `ensureBootstrappedValueSlotsOrThrow()`, `executeStep()`, trace state allocation, and teacher target upload. Only rows with `execution_active[b] == true` enter the execution path.
- Gated trace state tensor allocation (`Tensor::zeros`, `requires_grad_()`, `ensure_grad()`) to active rows only. Non-active rows keep default-constructed empty `trace_state_by_row[b]` — no GPU allocation, no grad tracking.
- Added per-row `execution_active[b]` gating on the cross-attention read loop: non-execution rows skip `crossAttentionRead()` entirely.
- Converted the silent bootstrap conditional in `Inference_GPU.cu` (`if (!slot_ptr || !ts.cached_token_numeric_values.data)` → skip bootstrap) to a hard `std::runtime_error` throw. Decode-time execution now always requires both slot map and numeric values; missing either is a fatal error.
- Simplified teacher target upload: removed redundant `execution_active[b]` check inside the loop body since only active rows enter the loop.

**Changed files**
- `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu`
- `resources/models/GRIM-text/training/Inference_GPU.cu`

**Ownership after change**
- `AutogradTraining.cu` owns the per-row execution activation decision using `execution_active[b]` from `BatchPayload`.
- `Inference_GPU.cu` unconditionally executes bootstrap when decode-time execution is active; missing prerequisites are fatal.
- `execution_block_memory_stream_GPU.cu`'s `ensureBootstrappedValueSlotsOrThrow()` remains the layer-internal fail-loud validator (unchanged, already correct from WS0).
- No code path silently skips execution for rows that are marked active.

**Non-responsibilities**
- WS7 does NOT change the `executeStep()` internal contract — the layer-internal throw from `ensureBootstrappedValueSlotsOrThrow()` was already correct.
- WS7 does NOT modify the shared validator (WS4) — validation runs before execution and catches structural metadata problems.

**Integration points / migrated consumers**
- `AutogradTraining.cu` per-row loop: all execution operations (memory allocate, bootstrap, K-step loop, teacher upload, cross-attention read) are gated behind `execution_active[b]`.
- `Inference_GPU.cu` decode path: bootstrap is unconditional when execution is active; the conditional skip path is deleted.

**Legacy deleted**
- Silent bootstrap skip in `Inference_GPU.cu` (`if (slot_ptr && ts.cached_token_numeric_values.data)` conditional that silently skipped bootstrap when data was missing)
- Implicit "all rows execute" behavior in `AutogradTraining.cu` row loop (non-execution rows previously entered `executeStep()` and were caught by layer-internal validation instead of being skipped at the orchestrator level)

**Validation**
- All 4 WS7 completion criteria now PASS:
  1. The orchestrator decides execution solely from compiled payload activation state (`execution_active[b]`) ✅
  2. Non-execution rows skip before `executeStep()` is called ✅
  3. Execution-active rows with empty bootstrap memory or no valid slot initialization throw immediately ✅
  4. No silent early-return path remains in `executeStep()` or in any split helper ✅
- Static audit: no remaining `if (slot_ptr &&` or `if (ts.cached_token_numeric_values` conditional-skip patterns in execution paths.
- Full CUDA compile validation is blocked on local macOS environment lacking CUDA toolchain.

**Flow diagram delta**
- Updated "Current artifact status" from "Workstream 7 next" to "Workstreams 3–7 complete; Workstream 8 next".
- Updated "Current enforced runtime flow" diagram: training row loop now shows explicit `execution_active[b]` gate before ExecutionBlock operations.

**Remaining gaps / next gate**
- Workstream 7 is complete. Proceed to Workstream 8: align validation path and training path so both call the same shared validator and fail on the same invalid rows.
