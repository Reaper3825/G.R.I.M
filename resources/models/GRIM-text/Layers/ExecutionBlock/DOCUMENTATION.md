# Execution Block — Documentation (v2.1)

This document describes the **Execution Block** (differentiable register machine), how it fits into training/inference, configuration, public APIs, data flow, and checkpoints.

> **Breaking change.** This version is a complete rewrite. Old (v1) checkpoints are **not loadable**; serialization hard-fails on schema mismatch.

**v2.1 doc update** (implementation alignment): differentiable context mean, no logit detach for arg softmax, atom-only decode MLP + row assembly, `inject_gate_temp`, all-`V` cross-attention with `valid_mask` gating, `num_ops > 0` / no sequential `num_filled` growth.

---

## What it does

The Execution Block runs **K sequential numeric steps** inside the encoder at a configurable layer. Each step:

1. Gathers a **unified candidate pool** (ScratchBlock atoms + filled memory slots).
2. Softmax-selects two operands (`p_arg1`, `p_arg2`) via learned scoring projections (logits stay on the autograd graph; masking is applied in-place before softmax).
3. Decodes scalars: **atom** candidates through a small MLP on embedding slices; **memory** candidates use stored slot values (`M.values`, gated by `valid_mask`) with **no** MLP on memory rows. Then computes a weighted combination of **`num_ops` arithmetic ops** (default **4**: `+`, `-`, `*`, `/`).
4. Embeds the scalar result via a learned linear projection.
5. **Injects** the result into one hidden row: `H[slot] += (1/sqrt(d_model)) * sigmoid((H[slot]·w_gate) * inject_gate_temp) * projected_result` (temperature-softened gate; not learnable).
6. Writes the result into **ExecutionMemory** using softmax-weighted blended slot updates.

Later encoder layers can also read memory through **gated cross-attention**.

### Hard rules (enforced in code)

- **No index-based selection** anywhere in execution logic. All selections (arg1, arg2, op, write slot) are softmax-weighted.
- **One sync per step** for numeric validation (`cudaStreamSynchronize` at end of each `executeStep`). No other host sync in the forward compute path.
- **Memory updates are weighted blending**, not overwrite.
- **Candidate gathering is differentiable** w.r.t. H via `GatherCandidateHiddenGradFn`. Gradients scatter-add back to H at atom positions, enabling the model to learn hidden representations useful for execution.
- **Candidate gathering uses masking** (`logits + (1 - mask) * (-1e9)` on **tracked** logits before softmax; memory slot values use `values * valid_mask` where applicable), not branching.
- **Context is `reduce_mean(H, dim=0)`**, shape `[1, d_model]`, implemented via a dedicated `kernelReduceMeanForward` kernel + `ReduceMeanGradFn` for differentiable backward (no avg_vec allocation; gradients broadcast `1/T` per row back into H).
- **Injection is scaled** by `1/sqrt(d_model)`.
- **Temperature** follows a defined annealing schedule applied to arg1, arg2, op, and write logits.
- **Execution steps sequentially update both H and M** — each step reads the output of the previous one.
- **Operations:** `config.num_ops > 0`; the fused kernel still implements the **first four** ops as `+`, `-`, `*`, `/` (division: `v1 / max(abs(v2), eps)` with `copysign`). Larger `num_ops` is reserved for future extension without dispatch tables yet.
- **Precision: fp32.** Masking sentinel is `-1e9`. fp16 is a future consideration.
- **Hard fail on NaN/Inf/magnitude.** Every step validates v1, v2, v_out, result_emb. Crashes if `isnan`, `isinf`, or `abs(x) > magnitude_limit` (default `1e6`).
- **Softmax validation.** Every softmax output (p_arg1, p_arg2, p_op, p_write) is validated: sum ≈ 1.0 (±1e-3), no NaN, no negatives. Violation throws.
- **CUDA_CHECK after every kernel.** All kernel launches and async memcpy calls are followed by `CUDA_CHECK` / `CUDA_CHECK_KERNEL()`.
- **Division clamp monitoring.** `kernelFourOps` increments a device counter when `abs(v2) <= eps` triggers the safe-division clamp. Readable via `lastDivClampCount(stream)`.
- **Atom embedding path is intentionally non-differentiable.** `kernelGatherCandidateAtomEmb` gathers from raw `float*` (ScratchBlock), not autograd tensors. Documented as constant input; gradients flow through the decode MLP weights only.
- **Injection grad_fn chaining.** `ExecutionBlockInjectGradFn::capture()` saves H's existing `grad_fn` before replacing it; backward chains to the saved parent, preserving the full upstream graph.
- **Fail-fast collapse (always fatal).** After each softmax: `p_arg1`, `p_arg2`, and `p_op` must have entropy ≥ `entropy_collapse_threshold` (default `0.01`); `p_write` must satisfy `max(p_write) ≤ write_collapse_threshold` (default `0.98`). Checked **before** downstream use. Same `d_numeric_error_flag_` as numeric/softmax errors; end-of-step sync throws `std::runtime_error` (no warn-and-continue).
- **debug_mode** only adds optional stderr echo of the fatal message before throw and enables **runtime metrics** (`ExecStepMetrics` in `ExecutionBlockStepOutput` when `diag_out` is provided). It does **not** relax validation.

---

## Signal priority

| Priority | Mechanism | Purpose |
|----------|-----------|---------|
| Primary | Injection into H | Result immediately influences next transformer layer |
| Secondary | Memory write (blended) | Long-term scratch storage |
| Tertiary | Cross-attention read | Downstream layers can attend to memory |

---

## Enabling the feature

### Requirements

1. **`execution_block_enabled = true`** in `LanguageModelConfig`.
2. **ScratchBlock enabled** — the autograd forward only schedules the Execution Block when `ctx.scratch_block && ctx.scratch_block->isEnabled()`. Without ScratchBlock, the block does not run (no atom pool).

### Training loss

Execution Block adds **entropy regularization** (not supervision CE):

```
L_entropy = -weight * sum_distributions( sum_i( p_i * log(p_i + eps) ) )
```

Applied to `p_arg1`, `p_arg2`, `p_op`, `p_write` distributions across all K steps. Prevents distribution collapse to one-hot. Weight controlled by `execution_block_entropy_weight` (default `0.01`).

---

## Configuration (`LanguageModelConfig`)

Fields in `grim_language_model_cuda.hpp`:

| Field | Default | Role |
|-------|---------|------|
| `execution_block_enabled` | `false` | Master switch |
| `execution_block_layer` | `-1` | Encoder layer index (resolved: `-1` → `num_layers - 2`, clamped to `[0, num_layers-1]`) |
| `execution_block_num_ops` | `4` | Must be `> 0`; default 4 matches the built-in `+`, `-`, `*`, `/` mix (see hard rules) |
| `execution_block_num_slots` | `4` | **V** — max memory slots |
| `execution_block_num_steps` | `2` | **K** — execution steps per forward |
| `execution_block_d_key` | `64` | Key dimension for memory addressing |
| `execution_block_d_type` | `8` | Per-slot type embedding width |
| `execution_block_cross_attn_head_dim` | `64` | Head dim for cross-attention read |
| `execution_block_cross_attn_topk` | `1` | Top-k keys per query |
| `execution_block_usage_decay` | `0.9` | Decay for slot usage after reads |
| `execution_block_diversity_kappa` | `2.0` | Penalty for rewriting the same slot |
| `execution_block_memory_slot_bias` | `0.5` | Bias against picking memory slots as args |
| `execution_block_temp_start` | `2.0` | Temperature at start of training |
| `execution_block_temp_end` | `0.5` | Temperature at end of training |
| `execution_block_temp_schedule` | `0` | `0` = linear, `1` = cosine |
| `execution_block_entropy_weight` | `0.01` | Weight for entropy regularization loss |
| `execution_block_diag_logging` | `false` | Log diagnostic metrics per step |

`ExecutionBlockConfig` in `execution_block_GPU.hpp` mirrors these fields plus decode MLP sizes (`value_decode_input_dim`, `value_decode_hidden_dim`), `empty_slot_bonus`, **`inject_gate_temp`** (default `0.5` — scales the pre-sigmoid dot product for injection; **not** a learnable parameter), **`deterministic`** (default `false` — reserved for fixed-seed / stable-order execution), **`debug_mode`** (default `false` — extra logging + `ExecStepMetrics` when diagnostics are enabled; **does not** change fail-fast behavior), **`entropy_collapse_threshold`** (default `0.01` — fatal if distribution entropy falls below), **`write_collapse_threshold`** (default `0.98` — fatal if `max(p_write)` exceeds), **`magnitude_limit`** (default `1e6` — fatal in `kernelCheckFinite` if any element exceeds `abs` bound), stream, and cuBLAS handle. Construction in `TrainingOps.cu` and `InitinferenceState.cu` maps `LanguageModelConfig` fields into `ExecutionBlockConfig` (if you add new config fields to `LanguageModelConfig`, wire them there; otherwise the C++ default applies).

---

## Temperature annealing

Temperature `T` is applied to all softmax selections:

```
p = softmax(logits / T)
```

Annealing schedule (linear, default):

```
T(t) = temp_start + (temp_end - temp_start) * (t / total_steps)
```

- **`T = 2.0` (start):** Soft, exploratory distributions — gradient flows evenly.
- **`T = 0.5` (end):** Sharp, decisive distributions — approaches hard selection.

Temperature is computed per-forward and passed to `executeStep`.

---

## Runtime placement (autograd forward)

Order inside the encoder loop (`AutogradTraining.cu`):

1. Run encoder layer `layer_idx` → `layer_output`.
2. If **`layer_idx == exec_layer`**: allocate/clear `exec_memory`, compute temperature `T`, then for **`step = 0 … K-1`** call `executeStep(layer_output, exec_memory, ..., T, ...)`.
   - Each step **mutates `layer_output` (H)** and **mutates `exec_memory` (M)** sequentially.
   - Step 2 reads the modified H from step 1, etc.
3. If **`layer_idx >= exec_layer`** and the execution block exists: call `crossAttentionRead(layer_output, exec_memory, ...)`.
4. Push `layer_output` into `encoder_layer_outputs`.

So: **write + inject** happens at `exec_layer` (K steps); **cross-attention read** happens at that layer and every deeper layer. Slot visibility in read is controlled by **`valid_mask`** (scores for near-empty slots are masked before softmax), not by a monotonic `num_filled` counter.

---

## Data flow per step

```
                 ┌─────────────────────┐
                 │  H [T, d_model]     │
                 │  M (ExecutionMemory) │
                 └──────┬──────────────┘
                        │
            ┌───────────▼────────────────┐
            │  1. Gather candidates       │
            │     atoms + memory slots    │
            │     → cand_hidden [C, dm]   │
            │     → cand_atom_emb [C, ae] │
            │     → valid_mask [C]        │
            └───────────┬────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  2. Decode values [C, 1]     │
            │     Atoms: MLP on slice     │
            │     Memory: M.values*mask   │
            │     Row-assemble (+GradFn)  │
            └───────────┬────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  3. Arg selection (softmax) │
            │     logits = w @ cand^T     │
            │     mask logits (tracked)   │
            │     p_arg = softmax(l / T)  │
            │     h_arg = p @ cand_hidden │
            │     v = p @ decoded_values  │
            └───────────┬────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  4. Op execution            │
            │     compute +,-,*,/ (all 4) │
            │     p_op = softmax(opL / T) │
            │     v_out = Σ p_op * result │
            │     (FourOpMixGradFn)       │
            └───────────┬────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  5. Linear value embedding  │
            │     result_emb = v_out *    │
            │       W_value_to_emb + b    │
            └───────────┬────────────────┘
                        │
          ┌─────────────┼─────────────┐
          │             │             │
  ┌───────▼───────┐  ┌──▼──────────┐  │
  │ 6a. Inject    │  │ 6b. Write   │  │
  │  one row:     │  │  p_write =  │  │
  │  gate(sig*τ)  │  │  softmax(.) │  │
  │ (InjectGradFn)│  │  blend M    │  │
  └───────────────┘  └─────────────┘  │
                                      │
                 ┌────────────────────┘
                 │
                 ▼
          H and M are mutated
          for next step
```

---

## Autograd strategy

The implementation uses **compositional autograd**: existing `autograd::` primitives (`matmul`, `softmax`, `concat`, `add`, `broadcast_add`, `silu`, `elementwise_mul`, …) handle most of the graph. **Four** custom `GradFn` nodes exist:

| GradFn | Scope | Why custom |
|--------|-------|-----------|
| `GatherCandidateHiddenGradFn` | Differentiable gather of H rows at atom positions into `[C, d_model]` | Backward scatter-adds gradients from candidate usage back to the original H rows via `kernelScatterAddHidden` (uses `atomicAdd` for duplicate positions). Chains to upstream H grad_fn. |
| `ReduceMeanGradFn` | `context = reduce_mean(H, dim=0)` → `[1, d_model]` | Forward sums H columns, backward broadcasts `grad / T` per row. Chains to upstream H grad_fn. |
| `DecodeAssembleGradFn` | Builds `[C,1]` from atom MLP output + raw memory scalars | Row-wise assembly is not a built-in op; backward routes the **first `num_atoms`** rows of `grad` to the atom decode chain only |
| `FourOpMixGradFn` | Weighted sum of op results | No `autograd::div` / fused safe-div; analytical gradients for `v1/max(abs(v2),eps)` live in `kernelFourOpMixBackward` |
| `ExecutionBlockInjectGradFn` | In-place add on one row of **H** with sigmoid gate × `inject_gate_temp` | In-place mutation + manual gate backward in `kernelInjectSlotBackward`. Chains to upstream H grad_fn (captures `H.grad_fn` before replacing). |

`kernelGatherCandidateAtomEmb` remains non-differentiable (atom embeddings originate from ScratchBlock raw pointers, not autograd tensors). The **decode MLP runs only on atom rows** and participates in the graph. **Context** uses `autograd::matmul` with a constant `1/T` averaging vector (that vector is not trained).

---

## Public API — `ExecutionBlockLayer`

Declared in `execution_block_GPU.hpp`, implemented in `execution_block_GPU.cu`.

### `executeStep(...)`

**Purpose:** One register-machine step. Softmax-selects args and op, computes scalar result, injects into H, blends into M.

```cpp
void executeStep(
    Tensor& H,                          // [total_tokens, d_model] mutated in place
    ExecutionMemory& M,                 // mutated in place
    const float* atom_embeddings,       // [num_atoms, atom_embedding_dim]
    const int* atom_positions,          // [num_atoms]
    int num_atoms,
    int total_tokens,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput* diag_out = nullptr  // optional diagnostics
);
```

`diag_out` is for logging only and must not drive computation. Contains detached copies of `p_arg1`, `p_arg2`, `p_op`, `p_write`, `v_out`, `result_emb`.

### `crossAttentionRead(...)`

```cpp
void crossAttentionRead(
    Tensor& hidden_states,              // [total_tokens, d_model] — updated in place
    ExecutionMemory& M,                 // usage updated
    int total_tokens,
    cudaStream_t stream
);
```

### `computeEntropyLoss(...)`

```cpp
Tensor computeEntropyLoss(
    const std::vector<ExecutionBlockStepOutput>& steps,
    float weight,
    cudaStream_t stream
) const;
```

Returns a scalar Tensor: `-weight * Σ_step Σ_dist H(p)` where `H(p) = -Σ p_i log(p_i + eps)`.

### Validation

All inputs are validated with `EXEC_CHECK` macros and `validateOrThrow` functions. Invalid shapes, null pointers, or out-of-range step indices cause `std::runtime_error` with context (file, line, message).

- `validateConfigOrThrow()`
- `validateMemoryOrThrow(M)`
- `validateExecuteStepInputsOrThrow(H, atom_embeddings, atom_positions, num_atoms, total_tokens, M, step)`
- `validateCrossAttentionInputsOrThrow(hidden_states, M, total_tokens)`

`validateConfigOrThrow()` requires `num_ops > 0` (not necessarily exactly 4).

---

## `ExecutionMemory` (register file)

Allocated via `ExecutionMemory::allocate(V, atom_dim, d_model, d_key, d_type, stream)` and cleared with `clear(stream)`.

| Member | Shape | Role |
|--------|-------|------|
| `values` | `[V, 1]` | Scalar stored per slot |
| `atom_embeds` | `[V, atom_dim]` | ScratchBlock-format encoding of result |
| `state_embeds` | `[V, d_model]` | Content vectors for cross-attention V |
| `valid_mask` | `[V]` | Probability-blended slot validity (0.0–1.0) |
| `usage` | `[V]` | Decayed usage from cross-attention reads |
| `write_score` | `[V]` | Learned overwrite preference bias |
| `key_embeds` | `[V, d_key]` | Addressing keys for write head and cross-attention |
| `type_embed` | `[V, d_type]` | Type tag (numeric) |
| `recent_write_mask` | `[V]` | Last-step write probability distribution |
| `num_filled` | `int` | Legacy field (initialized to 0); **not** incremented per step anymore. Cross-attention always uses **all V slots**; **do not** use `num_filled` to gate reads. |

### Memory blending (write)

Slot updates are **never hard-overwritten** in one shot; each field is **blended** with `p_write = softmax(write_logits / T)` over V slots:

```
M.values[v]       = (1 - p_write[v]) * M.values[v]       + p_write[v] * new_value
M.state_embeds[v] = (1 - p_write[v]) * M.state_embeds[v] + p_write[v] * new_embed
M.key_embeds[v]   = (1 - p_write[v]) * M.key_embeds[v]   + p_write[v] * new_key
M.valid_mask[v]   = max(M.valid_mask[v], p_write[v])
```

`valid_mask` is thus a running upper envelope of write mass; cross-attention masks slots whose `valid_mask` is ~0 so they do not receive attention weight.

---

## Learnable parameters

All registered under `ParamGroupType::EXECUTION_BLOCK`. 23 tensors total:

| Weight | Shape | Purpose |
|--------|-------|---------|
| `w_decode_1` | `[24, 16]` | Decode MLP layer 1 |
| `b_decode_1` | `[16]` | Decode MLP bias 1 |
| `w_decode_2` | `[16, 1]` | Decode MLP layer 2 |
| `w_arg1_select` | `[1, d_model]` | Arg1 scoring projection |
| `w_arg2_select` | `[1, d_model]` | Arg2 scoring projection |
| `W_op_select` | `[3*d_model, num_ops]` | Op logits from `[h_arg1, h_arg2, context]` (default `num_ops = 4`) |
| `W_key_proj` | `[d_model, d_key]` | Key generation from result embedding |
| `W_write_query` | `[4*d_model, d_key]` | Write query from `[h_arg1, h_arg2, context, result_emb]` |
| `W_write_key` | `[d_key, d_key]` | Write key scoring |
| `alpha` | `[1]` | Learned content score scalar (init 1.0) |
| `beta` | `[1]` | Learned usage penalty scalar (init 1.0) |
| `gamma` | `[1]` | Learned write score scalar (init 1.0) |
| `step_embeddings` | `[K, d_model]` | Per-step embedding |
| `type_num_embed` | `[d_type]` | Numeric type tag |
| `W_value_to_emb` | `[1, d_model]` | Linear value → embedding (replaces sinusoidal) |
| `b_value_to_emb` | `[1, d_model]` | Bias for value → embedding |
| `w_inject_gate` | `[d_model, 1]` | Dot-product weights for injection gate on the **result** token row |
| `W_Q_read` | `[d_model, head_dim]` | Cross-attention query |
| `W_K_read` | `[d_key, head_dim]` | Cross-attention key |
| `W_V_read` | `[d_model, head_dim]` | Cross-attention value |
| `W_O_read` | `[head_dim, d_model]` | Cross-attention output |
| `W_gate_read` | `[d_model, 1]` | Cross-attention per-token gate |
| `tau` | `[1]` | Cross-attention temperature (init 1.0) |

---

## Diagnostic logging

When `execution_block_diag_logging = true`, each `executeStep` call logs:

| Metric | Description |
|--------|-------------|
| `H(p_arg1)` | Entropy of arg1 selection distribution |
| `H(p_op)` | Entropy of op selection distribution |
| `abs(v_out)` | Magnitude of computed scalar result |
| `norm(projected_result)` | L2 norm of result embedding (before gate/scaling) |

Output goes to `stdout` via `kernelEntropy` and host-side reads (diagnostic only — not on forward critical path).

---

## Training integration — where to look in code

| Concern | Location |
|---------|----------|
| Encoder loop, `executeStep` / `crossAttentionRead` | `training/Autograd/AutogradTraining.cu` |
| Intermediates ownership | `training/Autograd/AutogradIntermediates.hpp` — `exec_memory`, `execution_block_output` |
| `AutogradContext` pointer | `training/Autograd/AutogradTraining.hpp` — `execution_block` |
| Context init | `initAutogradContext(...)` — call sites pass `model.getExecutionBlockLayer()` |
| Build layer | `training/TrainingOps.cu`, `Layers/InitInferenceState/InitinferenceState.cu` |
| Optimizer registration | `training/LanguageModel_Training.cu` — all 23 tensors under `EXECUTION_BLOCK` |
| Gradient norm bucket | `Shared/GradNorm/GradNormGPU.*`, `Phase2_TrainingLoop.cu` — `execution_block_*` |
| Entropy loss computation | `AutogradTraining.cu` — `computeEntropyLoss()` added to combined loss |
| Config fields | `GRIM/grim_language_model_cuda.hpp` |

---

## Checkpoints / FlatBuffers

Execution Block weights are stored as `ExecutionBlockWeights` inside `TransformerModel.execution_block` in the FlatBuffer schema (`grim_transformer_model_generated.h`).

### No backward compatibility

- **v1 checkpoints cannot be loaded.** The weight set, shapes, and count have changed.
- Loading performs **strict checks** on the presence and exact size of all 23 required weight vectors. Any mismatch is a hard failure (`std::runtime_error`).
- There is no fallback, no optional fields, no silent default initialization.
- Save/load path: `Layers/Serialization/Serialization_GPU.cu` packs/unpacks weight vectors; `Common/grim_model_serialization.cu` fills `SerializationSaveSources.execution_block` / `SerializationLoadRequest.execution_block`.

---

## CUDA kernels (execution_block_GPU.cu)

Gather / layout:

| Kernel | Purpose |
|--------|---------|
| `kernelGatherCandidateHidden` | Collect hidden states from atoms + memory into `[C, d_model]`; paired with `GatherCandidateHiddenGradFn` for differentiable backward |
| `kernelScatterAddHidden` | Backward kernel for differentiable gather: scatter-add gradients from `[C, d_model]` back to `[total_tokens, d_model]` at atom positions (`atomicAdd` for safety) |
| `kernelGatherCandidateAtomEmb` | Collect atom embeddings into `[C, atom_dim]` (non-differentiable) |
| `kernelBuildCandidateMask` | Build candidate mask `[C]` from atom count + memory `valid_mask` |
| `kernelSliceColumns` | Column slice of atom embeddings (used for **atom-only** decode input) |
| `kernelFillConstant` | Fill buffer with a scalar (e.g. `1/total_tokens` for differentiable mean context) |

Masking / assembly helpers:

| Kernel | Purpose |
|--------|---------|
| `kernelApplyLogitMask` | `logits += (1 - mask) * (-1e9)` (applied to **tracked** logits) |
| `kernelApplyValueMask` | `out[i] = values[i] * mask[i]` (memory scalar path into `[C,1]`) |

Arithmetic + mixing:

| Kernel | Purpose |
|--------|---------|
| `kernelFourOps` | Compute built-in op results: `v1+v2`, `v1-v2`, `v1*v2`, `v1/max(abs(v2),eps)`; increments `div_clamp_counter` when clamp triggers |
| `kernelFourOpMixForward` | `v_out = Σ_k p_op[k] * results[k]` |
| `kernelFourOpMixBackward` | Analytical gradients for `FourOpMixGradFn` (parameterized by `num_ops`) |

Injection (single result row):

| Kernel | Purpose |
|--------|---------|
| `kernelInjectResultSlot` | `H[slot] += (1/sqrt(d)) * sigmoid((H[slot]·w_gate)*τ) * result_emb` with `τ = inject_gate_temp` |
| `kernelInjectSlotBackward` | Backward for injection path (includes `inject_gate_temp` in sigmoid chain rule) |

Memory blending:

| Kernel | Purpose |
|--------|---------|
| `kernelBlendedWriteValues` | Blend scalar values into memory slots |
| `kernelBlendedWriteVectors` | Blend embedding vectors into memory slots |
| `kernelBlendedWriteValidMask` | `valid_mask[v] = max(valid_mask[v], p_write[v])` |

Cross-attention read (operates over **all V** slots; `valid_mask` gates attention mass):

| Kernel | Purpose |
|--------|---------|
| `kernelCrossAttnSharpScores` | Scores Q vs keys; slots with `valid_mask < 1e-6` get `-FLT_MAX` before softmax |
| `kernelCrossAttnWeightedValue` | Weighted sum of projected slot values |
| `kernelCrossAttnGatedOutput` | Apply `W_O` and per-token gate into `hidden_states` |
| `kernelDecayedUsageUpdate` | Decay + accumulate read attention into `M.usage` |

Diagnostics / validation:

| Kernel | Purpose |
|--------|---------|
| `kernelEntropy` | Compute `H(p) = -Σ p_i log(p_i + eps)` for a distribution |
| `kernelCheckFinite` | GPU scan for NaN/Inf/magnitude: `atomicMax(error_flag, stage_id)` if any element is non-finite or `abs(x) > magnitude_limit` |
| `kernelValidateSoftmax` | Validate softmax output: sum ≈ 1.0 (±1e-3), no NaN, no negative values |
| `kernelCheckEntropyCollapse` | Fatal if entropy &lt; threshold; `atomicMax` into `d_numeric_error_flag_` |
| `kernelCheckWriteCollapse` | Fatal if `max(p) &gt; threshold`; `atomicMax` into `d_numeric_error_flag_` |
| `kernelReduceMeanForward` | Forward: `out[j] = sum_i(H[i,j]) / T` for reduce_mean context |
| `kernelReduceMeanBackward` | Backward: `grad_H[i,j] += grad_out[j] / T` |
| `kernelComputeEntropyScalar` | Scalar entropy of a distribution to device buffer (metrics) |
| `kernelComputeMax` | Max element of array to device buffer (metrics) |

---

## Quick checklist for a new experiment

1. Enable ScratchBlock + Execution Block in config.
2. Set `execution_block_layer` (or rely on `-1` default for `num_layers - 2`).
3. Tune `execution_block_num_slots` (V) and `execution_block_num_steps` (K).
4. Set temperature schedule: `temp_start` (warm), `temp_end` (cool), and `temp_schedule`.
5. Adjust `entropy_weight` if distributions collapse too fast or too slow.
6. Tune **`inject_gate_temp`** in `ExecutionBlockConfig` (lower → softer injection gate; default `0.5`) if the gate saturates too early.
7. Enable `diag_logging` for debugging; disable for production runs.
8. Verify checkpoint save/load — old checkpoints **will not load**.
9. Use `getExecutionBlockLayer()` only when `execution_block_enabled` is true; otherwise `nullptr`.
