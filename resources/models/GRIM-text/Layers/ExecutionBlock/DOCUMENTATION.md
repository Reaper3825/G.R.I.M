# Execution Block — Current Implementation

This document describes the **current** `ExecutionBlock` implementation after Workstream 0 of the structured-execution cutover.

It is intentionally narrow:

- only the **live public surface** is documented,
- only the **current runtime behavior** is described,
- deleted kernels, deleted APIs, deleted config knobs, and deleted fallback paths are **not** preserved here for nostalgia.

## Current status

Workstream 0 leaves the ExecutionBlock in a much smaller and stricter shape:

- `execution_block_GPU.hpp` is the only public include.
- `execution_block_GPU.cu` is a thin public coordinator.
- private helpers are split into memory-stream and data-stream implementation files.
- `executeStep(...)` now requires **row-local** atom buffers and a **row-local** slot map.
- the layer no longer tolerates:
  - silent execution skip when no value slots were bootstrapped,
  - batch-global atom buffers passed into per-row execution,
  - decode-time `<NUM>` binding via last-write fallback.

Until the explicit decode-time slot selector exists, generation masks `<NUM>` instead of guessing a slot.

## File ownership

| File | Owns |
|---|---|
| `execution_block_GPU.hpp` | Public config, memory structs, diagnostics structs, `ExecutionBlockLayer` declaration |
| `execution_block_GPU.cu` | Validation helpers, construction/destruction, thin public wrappers |
| `execution_block_internal.hpp` | Private stage IDs, macros, `LayerAccess`, `StepWorkingSet` |
| `execution_block_memory_stream_GPU.hpp/.cu` | `ExecutionMemory` allocation/clear, bootstrap, slot reads/writes, recent-write bookkeeping, cross-attention read |
| `execution_block_data_stream_GPU.hpp/.cu` | Step-local data flow: context, trace encoding, arg/op/write distributions, result decode/injection, entropy loss |

`execution_block_internal.hpp` is private to `Layers/ExecutionBlock/`.

## Public surface

### `ExecutionMemory`

`ExecutionMemory` is the row-local register file used by the layer.

| Field | Shape | Meaning |
|---|---|---|
| `values` | `[V, 1]` | Scalar value stored per slot |
| `atom_embeds` | `[V, atom_embedding_dim]` | Atom-format embedding stored per slot |
| `state_embeds` | `[V, d_model]` | Hidden-state embedding stored per slot |
| `valid_mask` | `[V]` | Slot initialized / live mask |
| `usage` | `[V]` | Decayed cross-attention usage score |
| `key_embeds` | `[V, d_key]` | Addressing keys for write/read |
| `type_embed` | `[V, d_type]` | Slot type embedding |
| `recent_write_mask` | `[V]` | **One-hot** mask of the most recent hard write |

Public methods:

- `allocate(int V, int atom_dim, int d_model, int d_key, int d_type, cudaStream_t stream)`
- `clear(cudaStream_t stream)`

### `ExecutionBlockConfig`

`ExecutionBlockConfig` now contains **layer-owned behavior only**.
It does **not** carry orchestration-owned knobs such as execution-layer placement, temperature schedule, entropy schedule, stream ownership, or cuBLAS ownership.

Surviving fields:

- dimensions: `d_model`, `atom_embedding_dim`, `num_ops`, `num_slots`, `num_scratch_slots`, `num_exec_steps`, `value_decode_input_dim`, `value_decode_hidden_dim`, `d_key`, `d_type`, `cross_attn_head_dim`
- read/write behavior: `cross_attn_topk`, `usage_decay`, `empty_slot_bonus`, `diversity_kappa`, `inject_gate_temp`
- result placement: `result_slot_mode`, `result_slot_index`
- validation/debug gates: `debug_mode`, `entropy_collapse_threshold`, `write_collapse_threshold`, `magnitude_limit`, `transition_hard_threshold`

### `ExecutionBlockLayer`

Public methods:

- `executeStep(...)`
- `bootstrapMemoryFromSlotMap(...)`
- `crossAttentionRead(...)`
- `computeEntropyLoss(...)`
- validation helpers
- tensor accessors for registered parameters

Deleted public APIs such as `encodeState()` and `lastDivClampCount()` are gone.

## Row-local execution contract

`executeStep(...)` is explicitly row-local even though `H` is passed as the full `[total_tokens, d_model]` tensor.

The caller must provide:

- `token_offset` and `row_tokens` describing the active row span inside `H`,
- `token_to_slot_map` for that row only,
- `atom_positions` relative to `[0, row_tokens)`,
- non-null row-local atom buffers even when `num_atoms == 0`.

The layer fail-loud checks:

- `token_offset >= 0`
- `row_tokens > 0`
- `token_offset + row_tokens <= total_tokens`
- `token_to_slot_map != nullptr`
- `atom_embeddings != nullptr`
- `atom_positions != nullptr`
- every row-local atom position is in `[0, row_tokens)`
- every slot-bearing atom position maps to an initialized value-slot in `[S, V)`

Important current nuance:

- the boundary requires row-local atom buffers,
- validation uses row-local atom positions,
- but the **current step-local math reads operands only from value slots**.

So `atom_embeddings` are part of the enforced runtime contract today, but the current data-stream path does not yet consume them in operand selection.

## What one `executeStep(...)` does today

At a high level, one step does this:

1. **Validate memory bootstrap state**
   - fail if the execution-active row has no bootstrapped value slots.
2. **Optionally snapshot diagnostics**
   - capture pre-write `values` / `valid_mask` if `diag_out` is requested.
3. **Build the value-slot candidate set**
   - gather hidden/state/value tensors only from the value-slot range `[S, V)`.
4. **Build row-local context**
   - `context = reduce_mean(H[token_offset : token_offset + row_tokens])`
   - add encoded trace history from prior `ExecutionRecord`s.
5. **Select operands**
   - `p_arg1`, `p_arg2` are softmax distributions over value slots,
   - forward reads are materialized by hard argmax slot reads,
   - backward uses straight-through-style routing through `SlotValueSTGradFn`.
6. **Select and apply the operation**
   - compute the built-in four ops: `+`, `-`, `*`, safe `/`,
   - choose the forward result via hard argmax over `p_op`,
   - keep the soft distribution for diagnostics/loss/gradients.
7. **Update trace state**
   - encode the current discrete execution record,
   - update `trace_state` with a learned residual transform.
8. **Decode the scalar back into embeddings**
   - scalar → atom embedding slice encoding,
   - scalar → decode MLP → scalar,
   - scalar → `result_emb` via learned projection.
9. **Inject into hidden state**
   - inject `result_emb` into either the configured fixed token row or the last token of the row.
10. **Score write destinations**
    - compute `p_write` over all slots.
11. **Hard write back**
    - choose one slot by argmax over `p_write`,
    - overwrite `values`, `state_embeds`, `key_embeds`, `atom_embeds`, `type_embed`,
    - set `valid_mask[slot] = 1`,
    - set `recent_write_mask` to a one-hot vector for that slot.
12. **Run fail-loud post-step validation**
    - numeric checks,
    - softmax validity checks,
    - entropy / collapse checks,
    - optional transition-loss diagnostics,
    - multi-slot mutation check.

## Write semantics

The current implementation does **not** do blended writes.

It computes a softmax distribution `p_write` for learning/diagnostics, then performs a **hard argmax write** to exactly one slot in forward execution.

Consequences:

- `recent_write_mask` is a **one-hot** last-write marker,
- `valid_mask` is hard-set for the selected slot,
- multi-slot mutation is treated as a register-machine violation.

## Cross-attention read

`crossAttentionRead(...)` is owned by the memory-stream implementation.

Current behavior:

- project row-local hidden states into read queries,
- project all slots into keys/values,
- mask near-empty slots using `valid_mask`,
- optionally apply top-k sharpening,
- apply a learned read gate per token,
- update `usage` with decayed attention mass.

`crossAttentionRead(...)` is also row-local via `token_offset` / `row_tokens`.

## Diagnostics and fail-loud behavior

The layer carries device-side error/state buffers for runtime validation:

- numeric / stage error flag,
- division clamp counter,
- execution index scratch,
- packed execution-record scratch.

Softmax and numeric failures do **not** degrade gracefully.
They accumulate a stage id and `finalizeStepOrThrow(...)` throws at the end of the step with a named stage.

If `debug_mode` is enabled, the layer may emit additional stderr context before throwing, but it does not relax any validation rule.

## Custom autograd nodes currently in use

The current data-stream implementation owns these custom `GradFn`s:

- `ReluGradFn`
- `SlotValueSTGradFn`
- `FourOpMixGradFn`
- `ExecutionBlockInjectGradFn`
- `ReduceMeanGradFn`
- `RecordEncodeGradFn`
- `L1ScalarLossGradFn`

Removed or stale documentation about other grad nodes is intentionally gone.

## Learnable tensors

The layer currently owns these learnable tensor groups:

### Decode / projection

- `w_decode_1`, `b_decode_1`, `w_decode_2`
- `W_value_to_emb`, `b_value_to_emb`

### Operand / op selection

- `w_arg1_select`, `w_arg2_select`
- `W_op_select`

### Write path

- `W_key_proj`
- `W_write_query`, `W_write_key`
- `alpha`, `beta`

### Read path

- `W_Q_read`, `W_K_read`, `W_V_read`, `W_O_read`, `W_gate_read`, `tau`

### Trace / reasoning state

- `step_embeddings`
- `E_slot`, `E_op`, `W_scal`, `b_scal`, `W_trace`, `b_trace`, `W_reason_gate`

### Type / injection

- `type_num_embed`
- `w_inject_gate`

## What this layer does **not** own anymore

These concerns are outside the layer boundary now:

- execution-layer placement in the encoder stack,
- temperature scheduling over training steps,
- entropy-loss scheduling,
- stream or cuBLAS ownership,
- batch/global execution metadata validation,
- decode-time numeric slot selection policy.

## Inference note

Inference now follows the same strict row-local execution contract:

- decode-time ExecutionBlock calls build a ScratchBlock-backed row-local atom view,
- null atom-pointer decode calls are gone,
- the old last-write `<NUM>` binding fallback is gone,
- generation masks `<NUM>` until a dedicated decode-time selector exists.

That masking behavior is temporary in the overall cutover, but it is the **current enforced behavior**.
