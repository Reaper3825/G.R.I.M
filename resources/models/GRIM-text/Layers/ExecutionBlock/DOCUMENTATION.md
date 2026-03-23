# Execution Block — Documentation (v2)

This document describes the **Execution Block** (differentiable register machine), how it fits into training/inference, configuration, public APIs, data flow, and checkpoints.

> **Breaking change.** This version is a complete rewrite. Old (v1) checkpoints are **not loadable**; serialization hard-fails on schema mismatch.

---

## What it does

The Execution Block runs **K sequential numeric steps** inside the encoder at a configurable layer. Each step:

1. Gathers a **unified candidate pool** (ScratchBlock atoms + filled memory slots).
2. Softmax-selects two operands (`p_arg1`, `p_arg2`) via learned scoring projections.
3. Decodes scalar values and computes a weighted combination of **4 arithmetic ops** (`+`, `-`, `*`, `/`).
4. Embeds the scalar result via a learned linear projection.
5. **Injects** the result directly into hidden states: `H = H + (1/sqrt(d_model)) * gate * projected_result`.
6. Writes the result into **ExecutionMemory** using softmax-weighted blended slot updates.

Later encoder layers can also read memory through **gated cross-attention**.

### Hard rules (enforced in code)

- **No index-based selection** anywhere in execution logic. All selections (arg1, arg2, op, write slot) are softmax-weighted.
- **No CPU synchronization** (`cudaMemcpy`, `cudaStreamSynchronize`) during the forward pass.
- **Memory updates are weighted blending**, not overwrite.
- **Candidate gathering uses masking** (`logits + (1 - mask) * (-1e9)`, `values * mask`), not branching.
- **Context is `mean_pool(H)`** (mean over all tokens).
- **Injection is scaled** by `1/sqrt(d_model)`.
- **Temperature** follows a defined annealing schedule applied to arg1, arg2, op, and write logits.
- **Execution steps sequentially update both H and M** — each step reads the output of the previous one.
- **Operations restricted** to `+`, `-`, `*`, `/`. Division uses `v1 / max(abs(v2), eps)` with `copysign` for numerical stability.
- **Precision: fp32.** Masking sentinel is `-1e9`. fp16 is a future consideration.

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
| `execution_block_num_ops` | `4` | Fixed: `+`, `-`, `*`, `/` |
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

`ExecutionBlockConfig` in `execution_block_GPU.hpp` mirrors these fields plus decode MLP sizes (`value_decode_input_dim`, `value_decode_hidden_dim`), `empty_slot_bonus`, stream, and cuBLAS handle. Construction in `TrainingOps.cu` and `InitinferenceState.cu` maps `LanguageModelConfig` fields into `ExecutionBlockConfig`.

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
3. If **`layer_idx >= exec_layer`** and **`exec_memory.num_filled > 0`**: call `crossAttentionRead(layer_output, exec_memory, ...)`.
4. Push `layer_output` into `encoder_layer_outputs`.

So: **write + inject** happens at `exec_layer` (K steps); **cross-attention read** happens at that layer and every deeper layer.

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
            │  2. Decode values           │
            │     MLP: atom_emb → scalar  │
            │     → decoded_values [C]    │
            └───────────┬────────────────┘
                        │
            ┌───────────▼────────────────┐
            │  3. Arg selection (softmax) │
            │     logits = w @ cand^T     │
            │     mask logits + values    │
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
  │  H += scale * │  │  p_write =  │  │
  │  gate * proj  │  │  softmax(.) │  │
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

The implementation uses **compositional autograd**: existing `autograd::` primitives (`matmul`, `softmax`, `concat`, `add`, `mul`) handle most of the computation graph. Only two custom `GradFn` nodes exist:

| GradFn | Scope | Why custom |
|--------|-------|-----------|
| `FourOpMixGradFn` | Weighted sum of `+,-,*,/` results | No `autograd::div`, `autograd::abs`, or `autograd::reciprocal` available; analytical gradients for `v1/max(abs(v2),eps)` require a fused kernel |
| `ExecutionBlockInjectGradFn` | `H_out = H + (1/sqrt(d)) * gate * proj` | In-place mutation of H with scale factor; backward distributes grad to `gate` and `projected_result` |

Non-differentiable data preparation (`kernelGatherCandidateHidden`, `kernelGatherCandidateAtomEmb`, `kernelDecodeValues`) remains as raw CUDA kernels. Their outputs are detached inputs to the autograd graph — gradients do not flow through the decode MLP or initial gather.

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
| `num_filled` | `int` | Bookkeeping for read gating |

### Memory blending (write)

Slot updates are **never overwritten**. Given write probability `p_write = softmax(write_logits / T)` over V slots:

```
M.values[v]       = (1 - p_write[v]) * M.values[v]       + p_write[v] * new_value
M.state_embeds[v] = (1 - p_write[v]) * M.state_embeds[v] + p_write[v] * new_embed
M.key_embeds[v]   = (1 - p_write[v]) * M.key_embeds[v]   + p_write[v] * new_key
M.valid_mask[v]   = (1 - p_write[v]) * M.valid_mask[v]   + p_write[v] * 1.0
```

This ensures gradients flow through `p_write` for all slots.

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
| `W_op_select` | `[3*d_model, 4]` | Op logits from `[h_arg1, h_arg2, context]` |
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
| `w_inject_gate` | `[d_model, 1]` | Per-token injection gate |
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

Non-differentiable data prep kernels (raw CUDA, outputs are autograd-detached):

| Kernel | Purpose |
|--------|---------|
| `kernelGatherCandidateHidden` | Collect hidden states from atoms + memory into `[C, d_model]` |
| `kernelGatherCandidateAtomEmb` | Collect atom embeddings into `[C, atom_dim]` |
| `kernelBuildCandidateMask` | Build valid_mask `[C]` from atom count + memory valid_mask |
| `kernelDecodeValues` | MLP decode: atom embeddings → scalar values `[C]` |

Differentiable masking kernels:

| Kernel | Purpose |
|--------|---------|
| `kernelApplyLogitMask` | `logits += (1 - mask) * (-1e9)` |
| `kernelApplyValueMask` | `values *= mask` |

Arithmetic + mixing:

| Kernel | Purpose |
|--------|---------|
| `kernelFourOps` | Compute all 4 op results: `v1+v2`, `v1-v2`, `v1*v2`, `v1/max(abs(v2),eps)` |
| `kernelFourOpMixForward` | `v_out = Σ p_op[k] * results[k]` |
| `kernelFourOpMixBackward` | Analytical gradients for FourOpMixGradFn |

Injection:

| Kernel | Purpose |
|--------|---------|
| `kernelInjectResult` | `H[t] += (1/sqrt(d)) * gate[t] * projected_result` |
| `kernelInjectBackward_gate` | Gradient w.r.t. gate |
| `kernelInjectBackward_result` | Gradient w.r.t. projected_result |

Memory blending:

| Kernel | Purpose |
|--------|---------|
| `kernelBlendedWriteValues` | Blend scalar values into memory slots |
| `kernelBlendedWriteVectors` | Blend embedding vectors into memory slots |
| `kernelBlendedWriteValidMask` | Blend valid_mask toward 1.0 |

Diagnostics:

| Kernel | Purpose |
|--------|---------|
| `kernelEntropy` | Compute `H(p) = -Σ p_i log(p_i + eps)` for a distribution |

---

## Quick checklist for a new experiment

1. Enable ScratchBlock + Execution Block in config.
2. Set `execution_block_layer` (or rely on `-1` default for `num_layers - 2`).
3. Tune `execution_block_num_slots` (V) and `execution_block_num_steps` (K).
4. Set temperature schedule: `temp_start` (warm), `temp_end` (cool), and `temp_schedule`.
5. Adjust `entropy_weight` if distributions collapse too fast or too slow.
6. Enable `diag_logging` for debugging; disable for production runs.
7. Verify checkpoint save/load — old checkpoints **will not load**.
8. Use `getExecutionBlockLayer()` only when `execution_block_enabled` is true; otherwise `nullptr`.
