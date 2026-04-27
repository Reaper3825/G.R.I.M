---
name: PrecomputeBatchPayloads
overview: Phase1 authors all batch metadata and host payload values; Phase2 only selects the active batch by index at sync boundaries. Phase2 must not change any token-index to metadata mapping (remask, reslot, re-atomize, etc.). CPU must not rederive batch metadata in the hot loop; CUDA consumes already-authored state; only the batch index/descriptor selection changes. Enforce BatchPayload as immutable, POD-like, host-only (device buffers reusable/pre-resident, bindings outside the struct).
todos:
  - id: payload-contract
    content: Introduce BatchDeviceBindings (all d_* for the step); remove mutables from BatchPayload; computeLossBatch(const BatchPayload&, const BatchDeviceBindings&); no hidden state on TrainingState as implicit current batch
    status: completed
  - id: ctx-fields
    content: Add fixed schedule + payload vectors + epoch order to TrainingContext (Phase1_Startup.hpp)
    status: completed
  - id: planned-batches-module
    content: Create Startup/Batching/PlannedBatches module that builds fixed schedules and payload vectors in Phase1
    status: completed
  - id: phase1-wire
    content: Wire PlannedBatchesReady into Phase1_Startup.cu after PayloadBuildInputsReady
    status: completed
  - id: phase2-consume
    content: "Phase2: only active_batch = train_payloads[epoch_batch_order[epoch][i]] at sync; forbid buildBatchPayload, buildEpochBatches, shuffle train_views, rebuild catalog"
    status: completed
  - id: verify
    content: "Greps: forbidden Phase2 symbols; audit Phase2 for token↔metadata mutations (remask, slot rewrite, target mutation on payload/seq); compile/test if available"
    status: completed
isProject: false
---

# Precompute BatchPayloads in Phase1

## Goal
- Ensure **Phase2 never constructs batch runtime metadata** (no shuffling of `train_views`, no `buildEpochBatches`, no per-step `buildBatchPayload`).
- Phase1 must hand off a **fixed, fully prepared set of `GRIM::Batching::BatchPayload` objects** (train + val), plus **epoch ordering** (shuffle only the order of existing batches).
- Allow a narrow exception only if the ExecutionBlock truly needs per-step ephemeral state that cannot be precomputed.

## Why Phase1 owns all packing (CUDA rectangle vs. rebucketing)

`BatchPayload` is built against the **fixed training rectangle**: `RunCapacity` / model cache limits (`batch_rows`, `seq_cap`, token budget) and the **sliding-window** contract on sequence length. Padding is always to the **batch-local** `max_seq_len` within that cap, not a “free” reshape of the GPU problem.

**Consequence:** Per-epoch **repacking / rebucketing** does **not** improve the CUDA training rectangle — the device still sees a rectangle bounded by the same `max_cached_batch` × `max_cached_seq_len` (or equivalent). What changes is only **which rows sit in which batch** and **in what order**. That is **diversity of grouping**, not a more efficient **tensor geometry** for the kernels.

**Risk:** Rebucketing in Phase2 re-sews sequences that carry **token ↔ atom ↔ slot ↔ execution** metadata. Any mismatch in padding, slot maps, `teacher_steps`, or compiled execution payload across a new group membership is a **latent alignment bug** (hard to see, easy to “train wrong”). The work buys **row reordering for curriculum/diversity** at the cost of **re-executing** the full batch builder and re-validating invariants on every repack.

**Conclusion (hard product rule):** **Phase1 owns all packing** — one canonical schedule, prevalidated `BatchPayload` set. **Phase2 only increments or selects the active prevalidated batch** (indexing into Phase1 tables). Diversity across epochs, if desired, is **permutation of batch order** (`epoch_batch_order`), not rebuild of who is batched with whom, unless a future design explicitly re-runs full Phase1-style packing with the same validation gates.

## Hot loop, CPU vs CUDA, and sync-boundary invariants (explicit)

**What must be true in the training hot path**

- **CPU does not rederive batch metadata in the hot loop** — no padding math, no scheduler calls, no recomputation of `valid_tokens` / geometry / per-batch vectors from `TrainingSequence` at step time. All of that is authored in Phase1 (or in the single upload/synchronization slice immediately before the kernel, if the design keeps host tensors in the payload and still requires one staged copy to device buffers).
- **CUDA consumes already-authored payload state** — kernels read the device-resident copy that corresponds to the current batch. The **logical** batch is the Phase1-built `BatchPayload` (or equivalent descriptor + fixed host buffer layout); the GPU side is **either** pre-uploaded for each prebuilt batch **or** refilled from that immutable host snapshot at the **designated sync boundary** only, not re-derived from first principles in the loop.
- **Only the batch pointer / index changes at the sync boundary** — per step, Phase2’s job is to advance which batch is “active” (and possibly which device binding slot is active), not to recompute what a batch is.

**What Phase2 may do**

- **Increment or select the active batch descriptor** at the **designated synchronization point** (epoch advance, batch index advance, `cudaStreamSynchronize` / `cudaMemcpyAsync` completion before `computeLossBatch`, etc. — exact boundary is implementation detail, but it is a **small, explicit** region, not the body of the step).

**Hard invariant (allowed selection form)**

```cpp
const GRIM::Batching::BatchPayload& active_batch =
    ctx.train_payloads[ctx.epoch_batch_order[epoch][batch_i]];
```

(Equivalent: a single precomputed flat index `active_idx` if the implementation stores `epoch_batch_order` in another shape — the rule is still **indexing into a fixed Phase1 table**, not constructing a new `BatchPayload`.)

**Not allowed in Phase2 (forbidden; grep CI should fail if any remain)**

- `buildBatchPayload(...)` — construction happens only in Phase1 / builder during startup.
- `buildEpochBatches(...)` — schedule is fixed in Phase1; epoch order is `epoch_batch_order` only.
- `shuffle(ctx.data.train_views...)` — source view order and catalog for batching are fixed at startup; diversity is **batch order** only.
- `rebuild train_catalog` after startup — not in Phase2.
- **Any operation that changes token index ↔ metadata mapping** — includes remasking targets, rewriting slot maps, altering atom/execution fields, or reordering tokens in a way that changes which metadata applies to which position (see [Token index to metadata mapping](#token-index-to-metadata-mapping-hard-rule)). **Upload-only** copies that preserve byte-for-byte semantics are allowed at the sync boundary.
- **Mutating `TrainingSequence` or per-token side channels in Phase2** — not allowed; sequences are fixed after startup for the purpose of training consumption.

**Memory / loop concern (how this design addresses it)**

- If **`BatchPayload` is mostly logical descriptor + host tensor layout** (and device pointers are **not** hidden state on the struct, per the contract), then:
  - **Device buffers in `TrainingState` stay reusable and fixed-capacity** (`max_cached_batch × max_cached_seq_len`); the hot loop only **binds** the active host-authored batch to those buffers at the sync boundary, or uses **pre-uploaded** per-batch device mirrors if the plan stores them in Phase1.
- Precomputing N host `BatchPayload` values is a **RAM vs CPU** trade: it removes CPU work in the loop; if RAM is too high, a later phase can replace N full copies with **one in-place `BatchPayload` + Phase1 “recipe indices”** — but Phase2 still must **not** re-run `buildBatchPayload`; it would only **load** the next recipe into the single slot at the boundary.

## Token index to metadata mapping (hard rule)

- **Phase2 must not perform any operation that changes the mapping** between a **token index** (position in the batch’s flat `[batch * max_seq_len]` row-major layout) and its **associated metadata** — including but not limited to: `input_ids` / `target_ids`, `token_to_slot_map`, `atom_mask` / `atom_flags` / `atom_entry_ids`, `numeric_values`, `text_features`, `teacher_steps` / `teacher_step_mask`, `compiled_bootstrap_bindings`, `slot_selection_targets`, and any per-token execution supervision derived in `buildBatchPayload`. Those associations are **authored in Phase1** (or in the single builder that produces the `BatchPayload`); **Phase2 may only select which pre-authored batch is active**, copy/upload without semantic rewrite, and run the forward — never **re-mask**, **re-slot**, **re-atomize**, or **re-align** per-token data in the hot loop. If a mapping must change, that is a **data bug** fixed by regenerating the payload in **Phase1**, not a Phase2 step.

## BatchPayload contract (enforced — not aspirational)

**Intent:** `BatchPayload` is **host data only** describing one batch. It must be **immutable after `buildBatchPayload` returns**, **POD-like** in spirit (value semantics, no “surprise” state), and **no hidden or runtime-patched state** on the same object.

| Rule | Meaning |
|------|--------|
| **Immutable** | No non-`const` methods on `BatchPayload` except an explicit `validate` that only reads. After construction, no field may be written by Phase2, loss, or upload paths. |
| **POD-like** | Plain scalars + `std::vector` / `std::string` of **values**. Avoid “smart” shared ownership that blurs *who* mutates. Prefer `shared_ptr` only where unavoidable (e.g. `AtomTable`); if it stays, document it as the single intentional exception or replace with a stable `atom_table_id` + registry outside the payload. |
| **No hidden state** | **Remove `mutable` device pointer fields** from `BatchPayload` (`d_token_to_slot_map`, `d_atom_mask` in [BatchPayload.hpp](resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp)) — **insufficient** to “just move them” onto `TrainingState` without a **parallel, explicit binding type** (see below). |
| **No token↔metadata rewrite in Phase2** | Aligns with [Token index to metadata mapping](#token-index-to-metadata-mapping-hard-rule): after the builder runs, no code path in Phase2 may change which metadata belongs to which flat token index. |

### Parallel binding system: `BatchDeviceBindings` (required, not optional)

**Problem:** Deleting the `mutable` `d_*` fields only **moves** hidden state if upload code (e.g. [ComputeLossBatch.cu](resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu) lines 248–252 today) stashes device pointers on `TrainingState` and downstream reads an implicit “current batch” through opaque fields — that **recreates** the same class of bug with a different owner.

**Fix:** A **first-class** struct holds **all** device pointers the forward/loss path needs for this step, populated **only** in the H2D / sync slice from `BatchPayload` host arrays into the **reusable** `TrainingState` cache buffers. **Ownership** of the underlying device memory can remain in `TrainingState`; **naming the active pointers** is what must be explicit.

**Shape (audit against the upload path in `computeLossBatch` — extend if a kernel needs more):**

```cpp
struct BatchDeviceBindings {
    int*         d_input_ids;       // = cached_token_ids (after this batch’s upload)
    int*         d_target_ids;      // = cached_targets
    float*       d_numeric_values;  // = cached_token_numeric_values
    uint8_t*     d_atom_mask;      // = cached_token_atom_mask
    uint16_t*    d_text_features;  // = cached_token_text_features (0 if not used)
    uint32_t*    d_atom_flags;     // = cached_token_atom_flags (may be null if not allocated)
    int32_t*     d_token_to_slot_map; // = cached_token_to_slot_map
};
```

(Any additional H2D surfaces used by the same path — e.g. MTP or extra execution buffers — get fields here too so nothing is “invisible.”)

**API (explicit contract, no const lie):**

```cpp
float LanguageModel::computeLossBatch(
    const GRIM::Batching::BatchPayload& host_batch,
    const BatchDeviceBindings&          device);
```

- **`BatchPayload`:** immutable host description + host-side tensors (and execution metadata **on host** as today).
- **`BatchDeviceBindings`:** the only place **device** addresses for *this step* are read by autograd/forward; **not** stashed by mutating `const BatchPayload&`, and **not** smuggled as unnamed “current” pointers on `TrainingState` unless the bindings struct is the single reader-facing view.

`initAutogradContext` / `executeAutogradForward` should take **`device`** (or the pair) so kernels never reach for `payload.d_*` again.

**Enforcement (implementation phase):**
- Grep/audit: no writes to any `BatchPayload` field after the builder returns, except in the builder itself.
- No assignment to any `d_*` on payload; upload path **returns or fills** `BatchDeviceBindings` after `cudaMemcpyAsync` + sync.
- Grep: no new “implicit current batch” device pointers on `TrainingState` without a corresponding `BatchDeviceBindings` field and a single place that sets them.
- Optional hardening: `static_assert` or a `ConstHostBatch` alias for `BatchPayload` in the training loop.

**Order of work:** Tackle `payload-contract` in parallel with or **before** bulk vector storage of payloads in `TrainingContext`, so we do not persist thousands of `BatchPayload` objects that still carry hidden mutable device slots.

## Current state (evidence)
- Phase2 currently shuffles `ctx.data.train_views` and rebuilds `ctx.data.train_catalog`, then calls `GRIM::Batching::buildEpochBatches(...)` each epoch (see `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu` around the `shuffle_this_epoch` block and the `buildEpochBatches` call).
- `LanguageModel::computeLossBatch` takes a `const GRIM::Batching::BatchPayload&` and treats it as **single source of truth** (see `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`).
- `GRIM::Batching::buildEpochBatches` is explicitly per-epoch and includes RNG shuffle + RANDOM ordering (see `resources/models/GRIM-text/Shared/Batching/EpochBatching.cu`).

## Target data flow

```mermaid
flowchart TD
  Phase1[Phase1_Startup] -->|loads| Seqs[TrainingSequence(train,val)]
  Phase1 -->|builds_once| FixedSchedule[FixedTrainSchedule_BatchAssignmentVec]
  Phase1 -->|builds_once| TrainPayloads[TrainBatchPayloadVec]
  Phase1 -->|builds_once| ValSchedule[FixedValSchedule_BatchAssignmentVec]
  Phase1 -->|builds_once| ValPayloads[ValBatchPayloadVec]
  Phase1 -->|creates_each_epoch| EpochOrder[EpochOrderVec_of_batch_indices]

  Phase2[Phase2_TrainingLoop] -->|uses| EpochOrder
  Phase2 -->|indexes| TrainPayloads
  Phase2 -->|indexes| ValPayloads
  Phase2 -->|computeLossBatch| Model[LanguageModel]
```

## Design decisions
- **Rationale (CUDA rectangle + alignment):** see [Why Phase1 owns all packing](#why-phase1-owns-all-packing-cuda-rectangle-vs-rebucketing) — rebucketing does not change the device rectangle; it only reorders **row membership** and risks **token↔atom↔slot↔execution** misalignment, so all packing and validation is Phase1.
- **Fixed batch membership**: Build the train `BatchSchedule` once in Phase1; Phase2 only shuffles the order of batch indices per epoch.
- **Compartmentalized staging**:
  - **PlannedSequence**: already represented by `TrainingSequence` loaded from `.grmt` + post-processed by sliding window.
  - **PlannedBatch**: `BatchAssignment` + fully-built `BatchPayload`.
  - **PlannedEpoch**: vector of batch indices (per-epoch shuffle over existing `PlannedBatch` list).
- **ExecutionBlock exception**: if any field must be dynamic per step (rare), it must **not** rewrite **token index ↔ metadata** associations; only state that does not alter per-token alignment may live outside the prebuilt batch (see [Token index to metadata mapping](#token-index-to-metadata-mapping-hard-rule)).

## Implementation plan

### 1) Add Phase1-owned “planned batches” to `TrainingContext`
- Update `resources/models/GRIM-text/training/Phases/Phase1_Startup.hpp`:
  - Add fields for:
    - `GRIM::Batching::BatchSchedule fixed_train_schedule;`
    - `std::vector<GRIM::Batching::BatchPayload> train_payloads;`
    - `GRIM::Batching::BatchSchedule fixed_val_schedule;`
    - `std::vector<GRIM::Batching::BatchPayload> val_payloads;`
    - `std::vector<std::vector<int>> epoch_batch_order;` (or a compact representation: one vector reused and reshuffled per epoch seed)

### 2) Create a startup subsystem that builds the fixed schedule and payload vectors
- **Prerequisite:** [BatchPayload contract](#batchpayload-contract-enforced--not-aspirational) is satisfied: builder fills host fields only; device pointers and upload state live **outside** `BatchPayload` so each stored `BatchPayload` in `train_payloads` is a pure, replayable value.
- Add new startup module:
  - `resources/models/GRIM-text/training/Phases/Startup/Batching/PlannedBatches.hpp`
  - `resources/models/GRIM-text/training/Phases/Startup/Batching/PlannedBatches.cu`
- Responsibilities:
  - Build **one** `BatchSchedule` for train from `ctx.data.train_catalog` using capacity (`ctx.run_capacity.max_tokens_per_batch`, `ctx.run_capacity.batch_rows`).
    - For policy: reuse `GRIM::Batching::buildBatches(...)` directly with a Phase1 “fixed policy” (mirroring the current `EpochBatching` config but without epoch RNG dependence), or call `buildEpochBatches(... epoch=0 ...)` once and treat it as fixed.
  - Build `train_payloads` by iterating the schedule and calling `buildTrainPayload(ctx, assignment)` (which already uses Phase1-authored `ctx.payload_build_inputs`).
  - Build a fixed `val_schedule` via `GRIM::Batching::buildBatches(...)` (Phase2 currently does this inline in validation).
  - Build `val_payloads` similarly via `buildValPayload(ctx, assignment)`.
  - Produce `epoch_batch_order` (or an epoch RNG function) that yields permutations of `[0..train_payloads.size())` for each epoch.

### 3) Wire the new subsystem into Phase1 orchestration
- Update `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu` to call `PlannedBatchesReady(*ctx)` after:
  - Data is loaded + sliding window applied
  - `RunCapacity` is authored
  - Model is allocated
  - `PayloadBuildInputsReady(*ctx)` has run (so payload building uses the validated static inputs)

### 4) Refactor Phase2 to consume planned payloads only
- Update `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu` to match [Hot loop, CPU vs CUDA, and sync-boundary invariants](#hot-loop-cpu-vs-cuda-and-sync-boundary-invariants-explicit):
  - **Forbidden in Phase2:** `buildBatchPayload`, `buildEpochBatches`, `shuffle(train_views...)`, `rebuild train_catalog` (see that section for the full rule).
  - **Allowed:** only advancing batch / epoch and selecting the active payload per the hard invariant, e.g. `active_batch = ctx.train_payloads[ctx.epoch_batch_order[epoch][batch_i]]`.
  - Remove per-epoch shuffling of `ctx.data.train_views` and rebuilding `ctx.data.train_catalog`.
  - Remove calls to `GRIM::Batching::buildEpochBatches(...)`.
  - Hot loop: index into `ctx.epoch_batch_order` and `ctx.train_payloads`; after upload/sync, call `computeLossBatch(const BatchPayload&, const BatchDeviceBindings&)` with **no** metadata rebuild and **no** mutation of `BatchPayload` for `d_*`.
  - Validation: iterate `ctx.val_payloads` only (no `buildBatches` in the val loop).

### 5) Update logging + invariants
- Ensure the existing batching summary logging is emitted **once** during startup rather than every epoch.
- Add Rule-20 checks in the new startup module:
  - No empty payloads
  - All payloads `validate("startup")`
  - `payload.fits_in_cache == true` for every payload (this is already enforced by `validate`)

### 6) Compatibility + rollout
- Keep a temporary config escape hatch (optional): allow switching between fixed vs per-epoch batching for debugging.
  - This can be a hyperparameter (`fixed_batches_enabled`) or a compile flag.
  - Default should be **fixed** per your directive.

## Acceptance criteria
- **CPU:** no rederivation of batch metadata inside the **hot loop**; only **index/selection** of the pre-authored batch at the **sync boundary** (see [Hot loop...](#hot-loop-cpu-vs-cuda-and-sync-boundary-invariants-explicit)).
- **CUDA:** consumes already-staged or pre-authored device state; only the **active batch** changes step-to-step, not a rebuild of what the batch is.
- **Hard invariant in code** (train path): `active_batch` obtained only by indexing `ctx.train_payloads[ctx.epoch_batch_order[epoch][batch_i]]` (or an equivalent precomputed `active_idx` into `train_payloads`).
- **BatchPayload** matches the enforced contract: no `mutable` fields used for side effects; **all** per-step device pointers live in **`BatchDeviceBindings`**, not on payload or as an implicit “current” on `TrainingState` without the struct. Prebuilt `train_payloads`/`val_payloads` are value snapshots (host only).
- **Phase2 must not** call `buildBatchPayload`, `buildEpochBatches`, `shuffle` on `train_views`, or `rebuild train_catalog`.
- **Token mapping:** Phase2 performs **no** operation that changes **token index ↔ metadata** (see [Token index to metadata mapping](#token-index-to-metadata-mapping-hard-rule)); allowed: selection, byte-preserving upload, forward/loss — not remask/reslot/re-atomize/reorder semantics.
- **Grep/CI** can verify the forbidden symbols do not appear in `Phase2_TrainingLoop.cu` (or the Phase2 translation unit), except in dead `#if 0` (prefer zero exceptions).
- **`computeLossBatch(const BatchPayload&, const BatchDeviceBindings&)`** — second argument is **required**; host vs device split is the public contract (not optional polish).

## Key files to change
- `resources/models/GRIM-text/Shared/Batching/BatchPayload.hpp` (and [BatchPayload.cu](resources/models/GRIM-text/Shared/Batching/BatchPayload.cu)): remove `mutable` `d_*` from `BatchPayload`; align with POD-like + immutable host contract
- New (or colocated) **`BatchDeviceBindings`** definition — e.g. next to `BatchPayload.hpp` or in `Shared/Batching/`
- `resources/models/GRIM-text/Shared/Loss/ComputeLoss/ComputeLossBatch.cu`: H2D path fills `BatchDeviceBindings` from `TrainingState` cache after each batch’s copies; `computeLossBatch` / `initAutogradContext` take bindings, **never** write `payload.d_*`
- `LanguageModel` declaration(s) and all `computeLossBatch` call sites (Phase2, any tests): two-argument form
- Autograd / execution code that read `payload.d_token_to_slot_map` or `d_atom_mask`: switch to `BatchDeviceBindings` (or the pair)
- `resources/models/GRIM-text/training/Phases/Phase1_Startup.hpp`
- `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu`
- `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`
- New: `resources/models/GRIM-text/training/Phases/Startup/Batching/PlannedBatches.{hpp,cu}`

## Risks / trade-offs
- **Memory**: Prebuilding all `BatchPayload` vectors can be large. If it becomes too big, a later phase can use a single in-place host `BatchPayload` + Phase1 recipes; **device** side remains `BatchDeviceBindings` pointing at the **same** reusable `TrainingState` cache each step (not N× full GPU copies unless explicitly chosen).
- **Training dynamics**: Per-epoch rebucketing was only **row membership / order** diversity, not a better CUDA rectangle (see [Why Phase1 owns all packing](#why-phase1-owns-all-packing-cuda-rectangle-vs-rebucketing)). If new diversity is needed later, prefer **K precomputed Phase1 schedules** or **batch-order permutations**, not Phase2 repacking.
