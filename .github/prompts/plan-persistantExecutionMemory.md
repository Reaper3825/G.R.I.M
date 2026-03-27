---
name: Persistent execution trace
overview: Add TrainingState-owned host execution history (execution_trace_by_row) plus device temporal accumulator trace_state_by_row; fuse context via trace_vec (stateless history encoding) + trace_state (running sum of step_emb). Pattern B weights unchanged on ExecutionBlockLayer only.
todos:
  - id: trace-training-state
    content: Add execution_trace_by_row on TrainingState (host ExecutionRecord only); reset at forward start; append after each executeStep in AutogradTraining.cu
    status: pending
  - id: trace-inference-d2h
    content: Ensure record D2H when diag_out is null (optional out-param); copy/mirror trace before Inference_GPU autograd clear
    status: pending
  - id: trace-feedback-pattern-b
    content: Add record-encoding weights (slot/op embeddings + scalar projection + stack→d_model W_trace) on ExecutionBlockLayer; trace_vec = f(encoded ExecutionRecord history); H2D or device buffer of last N records per row; extend Serialization + schema version
    status: pending
  - id: trace-state-accumulator
    content: TrainingState trace_state_by_row[b] [1,d_model] zeros at forward start; executeStep after step_emb autograd::add into trace_state; optional RMSNorm if available; context' = context + trace_vec + trace_state; wire Tensor& in AutogradTraining per row/step—do not recompute trace_state from history
    status: pending
  - id: optional-decode-carry
    content: "Optional: ring buffer of last N ExecutionRecord across forwardStep for generation"
    status: pending
isProject: false
---

# Persistent execution trace and step feedback (Pattern B layout)

## Pattern B file pattern (repo convention)

Follow the same layout as **Embedding**, **LMHead**, **ReasoningHead**, and **ExecutionBlock** today:


| Concern                                                                                                                               | Where it lives                                                                                                               | Files                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Learnable weights** (encode `ExecutionRecord` → `d_model`: slot/op embeddings, scalar branch, `W_trace` stack→`d_model`, `b_trace`) | **Owned by the layer** — self-allocated in ctor, Xavier-init, serialized with the layer                                      | `[resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp)`, `[execution_block_GPU.cu](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu)`                            |
| **Forward / autograd math** for trace fusion                                                                                          | Same layer implementation (or a **sibling** `.cu` in the same folder only if the file becomes unwieldy)                      | Prefer extending `execution_block_GPU.cu`; if split: `ExecutionTraceFeedback_GPU.hpp` + `.cu` under `Layers/ExecutionBlock/` and **one new line** in `[TrainingLoop/CMakeLists.txt](resources/models/GRIM-text/training/TrainingLoop/CMakeLists.txt)` next to `execution_block_GPU.cu` |
| **Orchestration** (when to reset, append, pass prior tensors)                                                                         | Training loop / autograd forward                                                                                             | `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`                                                                                                                                                                                              |
| **Non-parameter runtime** (host `execution_trace_by_row`; device `trace_state_by_row`)                                                | `TrainingState` — same class of fields as `inference_exec_memory`, `kv_cache_len` (buffers/state, **not** trainable weights) | `[TrainingState_GPU.hpp](resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp)`                                                                                                                                                                                       |


**Do not** put trainable `Tensor` weights for trace encoding on `TrainingState` (that would violate Pattern B: parameters belong on self-managing layers, collected for optimizer/serialization via layer getters).

**Do not** put the persistent host `ExecutionRecord` list inside `AutogradIntermediates` if the goal is survival across `clear()`; keep host trace on `TrainingState` (or a small struct held by `TrainingState`).

---

## File separation of concerns

Single rule: **storage and batch lifecycle live in `TrainingState` + `AutogradTraining`; math, fusion, and learnable encoders live in `ExecutionBlockLayer`; persistence of new weights in Serialization + training param lists.** Nothing else gains new responsibilities.

| File | Owns / must implement | Must not |
|------|------------------------|----------|
| [`TrainingState_GPU.hpp`](resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp) (and [`TrainingStateGPU.cu`](resources/models/GRIM-text/Shared/TrainingState/TrainingStateGPU.cu) if you add ctor/zero helpers) | **Declarations** for `execution_trace_by_row`, `trace_state_by_row`; document lifetime (per-forward reset, device vs host). | Encoding logic, `executeStep` internals, optimizer registration, H2D of record fields beyond owning buffers the caller fills. |
| [`AutogradTraining.cu`](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu) | **Orchestration only:** at ExecutionBlock forward entry, resize/clear host trace + zero `trace_state_by_row`; for each `(b, step)`, call `executeStep` with pointers/refs to prior records (or packed device view), `Tensor& trace_state_by_row[b]`, existing `layer_output` / `M_b`; after return, `push_back(step_diag.record)` into `execution_trace_by_row[b]`. | Duplicating `trace_vec` / `step_emb` math; modifying `ExecutionMemory` or encoder layers. |
| [`execution_block_GPU.hpp`](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp) | **API surface:** extended `executeStep(...)` (e.g. `Tensor& trace_state`, inputs needed for prior-record encoding); **struct** `ExecutionRecord` (unchanged fields); declarations for trace-encoding weights (`E_slot_`, `E_op_`, `W_trace_`, etc.). | Batch sizing, `std::vector` resize for `B`, reading `TrainingState` globally. |
| [`execution_block_GPU.cu`](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu) | **All differentiable trace logic:** build `trace_vec` from encoded **prior** `ExecutionRecord` history; build `step_emb` from **current** step record; read/write **`trace_state`** via `autograd::add` (+ optional **RMSNorm**); form **`context' = context + trace_vec + trace_state`** and thread `context'` into existing concat/matmul paths; existing `kernelAssembleExecRecord` + diagnostics unchanged. | Owning `trace_state` storage (always passed in by reference); resetting per-batch state. |
| [`AutogradIntermediates.hpp`](resources/models/GRIM-text/training/Autograd/AutogradIntermediates.hpp) | **No change** for durable trace; keep `exec_memories` / `exec_outputs_per_row` as today. | Storing `execution_trace_by_row` or `trace_state_by_row` (they must survive `clear()`). |
| [`Inference_GPU.cu`](resources/models/GRIM-text/training/Inference_GPU.cu) | Optional: ensure records available / mirror host trace before `autograd_intermediates.clear()` if inference needs parity with training diagnostics. | Defining new inference managers; changing ExecutionBlock kernels. |
| [`LanguageModel_Training.cu`](resources/models/GRIM-text/training/LanguageModel_Training.cu) (and/or [`TrainingOps.cu`](resources/models/GRIM-text/training/TrainingOps.cu) where ExecutionBlock params are gathered) | **Register** new ExecutionBlock trace-encoding tensors for optimizer + grad paths. | Trace fusion implementation. |
| [`Serialization_save.cu`](resources/models/GRIM-text/Layers/Serialization/Serialization_save.cu) / [`Serialization_load.cu`](resources/models/GRIM-text/Layers/Serialization/Serialization_load.cu) + FlatBuffer schema if needed | **Persist** new layer weights only. | `trace_state` / `execution_trace_by_row` (runtime-only). |
| [`TrainingLoop/CMakeLists.txt`](resources/models/GRIM-text/training/TrainingLoop/CMakeLists.txt) | New `.cu` only if you split ExecutionBlock into a sibling file (default: **no** new translation unit). | N/A |
| [`ExecutionBlockTest.cu`](resources/models/GRIM-text/Tests/ExecutionBlockTest.cu) | Tests for shape/autograd smoke of extended `executeStep` if needed. | Full training loop duplication |

**Data flow across the boundary**

```text
AutogradTraining.cu                          ExecutionBlockLayer (execution_block_GPU.cu)
─────────────────────                        ─────────────────────────────────────────────
reset: execution_trace_by_row, trace_state_by_row[b] ← zeros
        │
        ├─ executeStep(H, M_b, …, trace_state_by_row[b], prior_records…)
        │       → uses trace_state IN; computes context', pool, …;
        │       → updates trace_state OUT (trace_state += step_emb)
        │
        └─ push_back(record) to execution_trace_by_row[b]   (host log; not used to recompute trace_state)
```

**Optional split (only if `.cu` size forces it):** move **only** trace-encoding helpers into `Layers/ExecutionBlock/ExecutionTraceEncode_GPU.hpp` + `.cu`, still **called solely** from `ExecutionBlockLayer`; **do not** introduce a separate top-level “trace manager” type.

---

## What already exists

You already have the per-step data on the wire; **use it as the feedback signal**, not `result_emb`:

```cpp
// ExecutionRecord (execution_block_GPU.hpp) — canonical step transcript
struct ExecutionRecord {
    int arg1_slot;
    int arg2_slot;
    int op_id;
    float value_before_1;
    float value_before_2;
    float value_after;
};
```

- `[kernelAssembleExecRecord](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu)` + host fill into `ExecutionBlockStepOutput::record` when `diag_out != nullptr`.
- `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)` runs K steps and stores `exec_outputs_per_row[b].steps` until `[AutogradIntermediates::clear()](resources/models/GRIM-text/training/Autograd/AutogradIntermediates.hpp)`.

---

## Part A — Persistent host trace (not autograd-owned)

1. On `TrainingState`: e.g. `std::vector<std::vector<ExecutionRecord>> execution_trace_by_row` (+ optional max steps).
2. **Also on `TrainingState` (additive, not a parameter):** `std::vector<Tensor> trace_state_by_row` — one tensor per row, shape `[1, d_model]`, **GPU**. At **start** of each full forward that runs ExecutionBlock: **resize** to `B`, **zero** each tensor (same reset as `execution_trace_by_row`). Persists **across the K execution steps** within that forward; cleared each forward (KV-cache style). **Do not** recompute `trace_state` from `execution_trace_by_row`; update it only via **`trace_state += step_emb`** inside `executeStep`.
3. After each `executeStep`, `push_back(step_diag.record)` for row `b`.
4. Leave `autograd_intermediates.clear()` unchanged for tensors/graphs.
5. Inference: ensure record D2H even when `diag_out` is null (small optional `ExecutionRecord* out` or always-on trace flag); copy trace before clear in `[Inference_GPU.cu](resources/models/GRIM-text/training/Inference_GPU.cu)` if you need it after forward.

---

## Part B — Encode prior steps (Pattern B weights on ExecutionBlockLayer)

**Shape constraint:** `W_op_select_` is `[3 * d_model, num_ops]` — keep `pool` at `3 * d_model`.

**Fix exactly — do not use `result_emb` history for trace feedback:**


| Wrong                               | Right                                            |
| ----------------------------------- | ------------------------------------------------ |
| `trace_vec = f(result_emb history)` | `trace_vec = f(encoded ExecutionRecord history)` |


**Approach:** residual fuse into `context`:

1. For step `t`, take the last up to `N` prior records for this row: `ExecutionRecord` tuples `(arg1_slot, arg2_slot, op_id, value_before_1, value_before_2, value_after)`.
2. **Encode each record** (learned, on the layer): e.g. `e1 = E_slot[arg1_slot]`, `e2 = E_slot[arg2_slot]` (or masked if invalid), `e_op = E_op[op_id]`, `e_val = W_scal @ [v1,v2,v_out]^T + b_scal` (with small `W_scal`), then `step_emb = fuse(e1, e2, e_op, e_val)` → `[d_model]` (concat + linear, or add + RMSNorm).
3. `trace_vec = matmul(flatten([step_emb_{t-N}, …, step_emb_{t-1}]), W_trace_) + b_trace_` → `[1, d_model]` (pad missing steps with zeros). **Unchanged:** `trace_vec` remains a **stateless** encoding of **prior** record history only (`what happened`).
4. **`trace_state` (temporal accumulator, additive):** before forming `context'`, read `trace_state_by_row[b]` (running state at start of step `t`). After this step’s `step_emb` is produced from the **current** step’s encoded `ExecutionRecord`, update **`trace_state = trace_state + step_emb`** with existing **`autograd::add`** (differentiable; **no new CUDA kernels**). Optionally apply **`RMSNorm`** on `trace_state` if an existing autograd helper matches global stack usage. **Do not** derive `trace_state` by summing history from `execution_trace_by_row` each time—that would duplicate the role of `trace_vec`.
5. `context' = context + trace_vec + trace_state` (same `[1, d_model]` residuals throughout — **no** change to `W_op_select_` input width `3 * d_model`). Optional **RMSNorm** on `context'` only if already consistent with the rest of the block.

Prior records used for `trace_vec` are **constants for that forward pass** (history from steps `0..t-1`); gradients train the **encoders** and `W_trace_` / `b_trace_` against the current step’s loss. Feed encoded history via **H2D** of a packed float buffer built on host from `execution_trace_by_row[b]` / prior `step_diag.record`s, or a preallocated device ring updated after each step.

**Ordering inside `executeStep` (integrate into existing flow):** build `trace_vec` from prior records → load `trace_state` for row → form `context' = context + trace_vec + trace_state` and use `context'` everywhere `context` previously fed the op/write path → after the step’s `ExecutionRecord` is known, compute `step_emb` from that record → **`trace_state = trace_state + step_emb`** and write back to `trace_state_by_row[b]` for the next step.

**Implementation location (Pattern B):**

- Declare `E_slot_`, `E_op_`, `W_scal_`/`b_scal_` (or equivalent), `W_trace_`, `b_trace_` on `ExecutionBlockLayer` in `execution_block_GPU.hpp`.
- Allocate + Xavier-init in the layer constructor in `execution_block_GPU.cu`.
- Implement `encodeRecordHistoryToTraceVec(...)` in the same `.cu` (kernel or small autograd matmul chain from a `[N, d_rec]` or flattened buffer).
- Expose new tensors to serialization via existing ExecutionBlock save/load paths (`[Serialization_save.cu](resources/models/GRIM-text/Layers/Serialization/Serialization_save.cu)` / `[Serialization_load.cu](resources/models/GRIM-text/Layers/Serialization/Serialization_load.cu)`); bump FlatBuffer / version if required.
- `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`: pass the row’s prior `ExecutionRecord`s (or device buffer) for `trace_vec` as already planned; **additionally** pass **`Tensor& trace_state`** (i.e. `trace_state_by_row[b]`) into `executeStep` so the layer can add `step_emb` after each step. Reset `trace_state_by_row` to zeros when resetting `execution_trace_by_row`.

**Optimizer / param list:** register all new encoding weights wherever ExecutionBlock parameters are already enumerated (e.g. `[LanguageModel_Training.cu](resources/models/GRIM-text/training/LanguageModel_Training.cu)` / training ops).

**Non-goals (accumulator):** no changes to `ExecutionMemory`, slot kernels, `ExecutionRecord` layout, `AutogradIntermediates` contract, or inference recompute path; **skip serializing** `trace_state` (runtime-only, reset per forward).

---

## Optional — Cross-token decode carry

Part B covers **within-forward** K steps. For **across** `forwardStep` during generation, add a `TrainingState` ring buffer of last N `ExecutionRecord` (runtime only), uploaded each step — still no weights on `TrainingState`.

---

## Risks

- Checkpoint compatibility for new ExecutionBlock tensors.
- Inference: record path when `diag_out` is nullptr.
- `trace_state_by_row[b]` must stay on the autograd path when updated (`autograd::add`); assign the result tensor back into the vector so step `t+1` sees the chained graph.

---

## Tests

- `[ExecutionBlockTest.cu](resources/models/GRIM-text/Tests/ExecutionBlockTest.cu)` + short training smoke after wiring.

---

## Acceptance (additive)

1. Step-to-step dependency: later steps see **`trace_state`** carrying **`sum step_emb`**, not only bagged `trace_vec`.
2. Intra-forward persistence: **`trace_state`** evolves across K steps; reset per forward with **`execution_trace_by_row`**.
3. No regression: **`W_op_select_`** input remains **`3 * d_model`**; **`execution_trace_by_row`** and **`trace_vec = f(history)`** logic stay as specified above.
4. Scope: **`ExecutionBlockLayer` + `TrainingState`** only for this addition; no new global managers.

