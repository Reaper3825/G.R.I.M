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
- **Active workstream:** Workstream 0 — execution_block file deflation before semantic cutover
- **Last completed gate:** None
- **Next gate:** Finish Workstream 0 caller/build/doc reconciliation and validate the split build
- **Current implementation posture:** Initial Workstream 0 slice landed; public layer surface trimmed and the first private split files now exist

## Workstream-specific update contract

### Workstream 0 — execution_block file deflation before semantic cutover

**Status**
- In progress

**Each update must record**
- deleted kernels, deleted APIs, and deleted `ExecutionBlockConfig` surface
- final public/private include boundary after each split step
- which helpers moved into memory-stream vs data-stream ownership
- whether any silent skip, batch-global atom adaptation, or invalid decode-time `<NUM>` fallback path was removed
- validation performed on the surviving layer surface

### Workstream 1 — canonical structured execution source-of-truth model

**Status**
- Not started

**Each update must record**
- semantic types added or moved into `ExecutionMetadata.hpp`
- which files now own compiled payload fields
- the authoritative execution-active signal after the change
- any `D_row` or `R` contract clarification introduced
- validation or compile impact from metadata shape changes

### Workstream 2 — canonical structured sequence builder replaces `__SLOTS__`

**Status**
- Not started

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
