# Execution Block — Current Implementation

This document describes the **current** `ExecutionBlock` implementation after Workstream 6 of the structured-execution cutover.

It is intentionally narrow:

- only the **live public surface** is documented,
- only the **current runtime behavior** is described,
- deleted kernels, deleted APIs, deleted config knobs, and deleted fallback paths are **not** preserved here for nostalgia.

## Current status

Workstream 0 leaves the ExecutionBlock in a much smaller and stricter shape:

- `execution_block_GPU.hpp` is the only public include.
- `execution_block_GPU.cu` is a thin public coordinator.
- private helpers are split into memory-stream and data-stream implementation files.
- `executeStep(...)` sources atom positions directly from the **global** atom mask (`bindings.d_atom_mask`) row slice, and uses a **row-local** slot map.
- the layer no longer tolerates:
  - silent execution skip when no value slots were bootstrapped,
  - batch-global atom buffers passed into per-row execution,
  - decode-time `<NUM>` binding via last-write fallback.
- numeric atom placeholders remain real learned tokens. Decode-time generation may emit them only when selector/runtime state can bind a concrete slot/value; if that state is unavailable, inference masks numeric atom placeholders instead of guessing a slot.

## File ownership

| File | Owns |
|---|---|
| `execution_block_GPU.hpp` | Public memory structs, `ExecutionBlockDiagnosticsBuffers`, free execution-block op declarations |
| `execution_block_GPU.cu` | Diagnostics buffer allocation/destruction and thin public wrappers |
| `execution_block_internal.hpp` | Private stage IDs, macros, `StepWorkingSet` |
| `execution_block_memory_stream_GPU.hpp/.cu` | `ExecutionMemory` allocation/clear, bootstrap, slot reads/writes, recent-write bookkeeping, cross-attention read |
| `execution_block_data_stream_GPU.hpp/.cu` | Step-local data flow: context, trace encoding, arg/op/write distributions, result decode/injection, entropy loss |

`execution_block_internal.hpp` is private to `Layers/ExecutionBlock/`.

## Public surface

### `ExecutionMemory`

`ExecutionMemory` is the row-local register file used by the execution-block math.

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

### `HyperParameters::ExecutionBlockConstructionHP`

The public execution-block ops consume `HyperParameters::ExecutionBlockConstructionHP` explicitly.
There is no layer-local compatibility config wrapper and no runtime shell class anymore. Runtime handles such as CUDA stream and cuBLAS ownership remain explicit per-call inputs, not hyperparameters.

Grouped fields consumed by the execution-block ops:

- dimensions: `d_model`, `atom_embedding_dim`, `num_ops`, `num_slots`, `num_scratch_slots`, `num_exec_steps`, `value_decode_input_dim`, `value_decode_hidden_dim`, `d_key`, `d_type`, `cross_attn_head_dim`
- read/write behavior: `cross_attn_topk`, `usage_decay`, `inject_gate_temp`
- result placement: `result_slot_mode`, `result_slot_index`
- validation/debug policy: `debug_mode`, `magnitude_limit`, and `transition_hard_threshold` control fail-loud state/numeric checks; `entropy_collapse_threshold` and `write_collapse_threshold` are warning thresholds for trainable selection confidence

### `ExecutionBlockDiagnosticsBuffers`

`ExecutionBlockDiagnosticsBuffers` is the explicit runtime-owned device workspace passed into `executionBlockStep(...)`.

Public methods / accessors:

- `allocate(cudaStream_t stream)`
- `destroy()`
- `allocated()`
- `numericErrorFlag()`
- `divClampCount()`
- `divInvalidFlag()`
- `execIndices()`
- `execRecordI()`
- `execRecordF()`
- `reinforceBaseline()`

Ownership note:

- the first six buffers are per-step execution workspace / diagnostics,
- `reinforceBaseline()` is the durable REINFORCE EMA carried on the runtime owner.

### Execution-block free operations

Public methods:

- `executionBlockStep(...)`
- `executionBlockBootstrapMemoryFromSlotMap(...)`
- `executionBlockCrossAttentionRead(...)`
- `executionBlockComputeEntropyLoss(...)`

Deleted public APIs such as `ExecutionBlockLayer`, `prepareForwardRuntime(...)`, the public validation helpers, `encodeState()`, and `lastDivClampCount()` are gone.

The math entry points that consume trainable tensors (`executionBlockStep(...)`,
`executionBlockBootstrapMemoryFromSlotMap(...)`, and
`executionBlockCrossAttentionRead(...)`) receive the registry-owned
`ExecutionBlockParameterTensors` bundle explicitly. Runtime scratch is passed
explicitly through `ExecutionBlockDiagnosticsBuffers` where needed.

## Row-local execution contract

`executeStep(...)` is explicitly row-local even though `H` is passed as the full `[total_tokens, d_model]` tensor.

The caller must provide:

- `token_offset` and `row_tokens` describing the active valid row span inside `H` (`payload.seq_lengths[b]`, not padded `payload.max_seq_len`),
- `token_to_slot_map` for that row only.

Numeric-atom annotations are read directly from the global atom mask
(`bindings.d_atom_mask`) row slice. State-bearing positions are not inferred from
that mask: they are defined exclusively by the row-local `token_to_slot_map`, which
is validated against `compiled_bootstrap_bindings`. `payload.atom_mask` is validated
at build time to equal `token_layout.isAtom(token_id)`, so it matches ScratchBlock's
atom detection exactly.

The execution-block boundary fail-loud checks:

- `token_offset >= 0`
- `row_tokens > 0`
- `token_offset + row_tokens <= total_tokens`
- `token_to_slot_map != nullptr`
- `bindings.d_atom_mask != nullptr`
- every mapped position (`token_to_slot_map[pos] >= 0`) is a numeric atom and maps to an initialized value-slot in `[S, V)`
- ordinary numeric atoms may remain unmapped (`token_to_slot_map[pos] == -1`); they are not execution state

Important current nuance:

- the global atom mask identifies numeric tokens and excludes them from decision-context pooling,
- the compiled slot map identifies the strict subset that seeds execution state,
- but the **current step-local math reads operands only from value slots**.

`atom_embeddings` were removed from the `executeStep()` contract (WS6) because the data-stream path does not consume them.

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
   - inject `result_emb` into the row-final valid token only.
   - fixed result-slot mode is legal only when the configured slot equals that row-final absolute token index.
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
    - multi-slot mutation check.

## Shared-forward runtime preparation

Shared forward now routes execution-layer reset through one explicit helper:

- `Forward::provisionExecutionForwardRuntime(...)`

This helper prepares caller-owned execution runtime for one forward execution boundary.
It does not own the runtime; the caller still passes the actual storage explicitly:

- `std::vector<ExecutionMemory>& exec_memories`
- `std::vector<Forward::ExecutionBlockOutput>& exec_outputs_per_row`
- `std::vector<std::vector<Forward::ExecutionRecord>>& execution_trace_by_row`
- `std::vector<Tensor>& trace_state_by_row`

`Forward::ExecutionBlockOutput`, `Forward::ExecutionBlockStepOutput`,
`Forward::ExecutionRecord`, and `Forward::ExecStepMetrics` are declared in
`Shared/Forward/ModelForwardOutputs.hpp` because they are forward-owned Category 1
sink payloads even though execution-block free ops populate them. Durable
`ExecutionBlockParameterTensors` do NOT live there; they are owned by
`StartupParameterRegistry`, initialized by
`ParameterGroupRegistration::initializeExecutionBlockParameterTensors(...)`,
and passed explicitly into execution-block math entry points.

Current `provisionExecutionForwardRuntime(...)` behavior:

1. validates config + payload execution geometry,
2. resizes the caller-owned execution bags to `payload.batch_size`,
3. clears prior execution traces and step diagnostics,
4. allocates + zeroes each active row's `ExecutionMemory`,
5. recreates each active row's `trace_state`, enabling autograd only when the caller requested a connected parameter graph.

Durable execution-block diagnostics live alongside that runtime on
`Forward::ModelForwardExecutionRuntime::execution_diag`; shared forward calls
`ensureDiagnostics(stream)` separately so the REINFORCE baseline survives across
steps while per-row Category 1 execution bags are re-provisioned.

This keeps execution cleanup behind one fail-loud execution-block boundary instead of scattering vector clears and `ExecutionMemory::clear(...)` calls through shared forward.

## Write semantics

The current implementation does **not** do blended writes.

It computes a softmax distribution `p_write` for learning/diagnostics, then performs a **hard argmax write** to exactly one slot in forward execution.

The content term uses scaled query/key scoring, `dot(q_write, K_write) / sqrt(d_key)`, before the learned content coefficient is applied. This keeps initialization variance independent of key width and avoids saturating the write softmax before structured cross-entropy can train it.

Low selection entropy or `max(p_write)` above the configured confidence threshold is diagnostic, not an invalid execution state. Debug mode reports those conditions and training continues; non-finite or non-normalized softmax output remains fatal. Entropy regularization belongs to the auxiliary loss rather than the execution-state validity flag.

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

In shared autoregressive forward, the row's available `ExecutionMemory` is the **row-final** post-execution register state. It is not timestep-aligned. The caller must therefore read it back only at the row-final valid token (`token_offset = b * max_seq_len + seq_lengths[b] - 1`, `row_tokens = 1`) and only on the **next layer input or later**. The execution layer may export the immediate step result directly from `executeStep(...)`, but persistent register memory should not be written and then immediately consumed on the same layer boundary. Reading that same memory into earlier positions would leak future numeric atoms through the execution side channel.

## Diagnostics and fail-loud behavior

The runtime carries device-side error/state buffers for validation through
`ExecutionBlockDiagnosticsBuffers`:

- numeric / stage error flag,
- division clamp counter,
- division-invalid flag,
- execution index scratch,
- packed execution-record scratch,
- persistent REINFORCE baseline.

Softmax and numeric failures do **not** degrade gracefully.
They accumulate a stage id and `finalizeStepOrThrow(...)` throws at the end of the step with a named stage.

If `debug_mode` is enabled, the layer may emit additional stderr context before throwing, but it does not relax any validation rule.

## Custom autograd nodes currently in use

The current data-stream implementation owns these custom `GradFn`s:

- `SlotValueSTGradFn`
- `ExecutionBlockInjectGradFn`
- `ReduceMeanGradFn`
- `RecordEncodeGradFn`
- `SelectionCrossEntropyGradFn` — attached to teacher-supervised selection CE tensors when `selection_targets->valid`

Removed or stale documentation about other grad nodes is intentionally gone.

## Learnable tensors

The runtime layer no longer owns learnable tensors. These tensor groups are
durably owned by `StartupParameterRegistry::execution_block_parameters`,
initialized by `ParameterGroupRegistration`, and consumed explicitly by shared
forward / autograd call sites:

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

## What the execution-block subsystem does **not** own anymore

These concerns are outside the execution-block math boundary now:

- execution-layer placement in the encoder stack,
- temperature scheduling over training steps,
- entropy-loss scheduling,
- stream or cuBLAS ownership,
- batch/global execution metadata validation,
- decode-time numeric slot selection policy.

## Inference note

Inference now follows the same strict row-local execution contract:

- KV-cached inference prefill may run the gate and structured execution steps,
- the final prefill register file moves into session-owned `GenerationState`,
- cached decode windows borrow that memory for readback at layers after the execution layer,
- decode windows do not re-run the gate, bootstrap, or execution steps,
- semantic register values and validity remain unchanged during decode readback (`usage` remains telemetry),
- a model-confirmed terminal STOP step exposes its valid finite write value as the pending generation result,
- max-step termination and arbitrary live-slot recency never select a rendered result,
- the pending result is materialized as a session-owned AtomTable entry when the LM emits the matching numeric placeholder,
- prefill execution sources atom positions from the global atom mask (`bindings.d_atom_mask`),
- null atom-pointer execution calls are forbidden,
- the old last-write `<NUM>` binding fallback is gone,
- generation masks numeric atom placeholders only when decode-time selector/runtime state cannot supply a concrete slot binding.

That conditional masking behavior is the **current enforced inference behavior**.
