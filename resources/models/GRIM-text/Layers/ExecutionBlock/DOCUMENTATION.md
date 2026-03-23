# Execution Block — documentation

This document describes the **Execution Block** (differentiable register machine), how it fits into training/inference, configuration, public APIs, data flow, and checkpoints.

---

## What it does

The Execution Block runs **K discrete numeric steps** inside the encoder: it selects two operands from a **unified candidate pool** (ScratchBlock atoms + filled memory slots), decodes scalars, applies **one** of eight arithmetic ops via **hard argmax + STE**, and writes the result into **ExecutionMemory** (not into token hidden states). Later encoder layers **read** memory only through **gated cross-attention**.

**Design constraints (still enforced in code):**

- No direct injection of execution results into the main residual stream except through the cross-attention read path.
- Ops and discrete choices use STE, not soft mixtures over op outputs.
- Memory is explicit (`ExecutionMemory`); chaining happens across steps `0 … K-1` within one forward.

---

## Enabling the feature

### Requirements

1. **`execution_block_enabled = true`** in `LanguageModelConfig`.
2. **ScratchBlock enabled** — the autograd forward only schedules the Execution Block when `ctx.scratch_block && ctx.scratch_block->isEnabled()`. Without ScratchBlock, the block does not run (no atom pool).

### Optional: auxiliary alignment loss

If **`reasoning_head_enabled`** is also true, training adds a small **supervision term**: cross-entropy of Execution Block logits (op + arg1 + arg2 per step) against **argmax targets** taken from the Reasoning Head’s own logits. This bootstraps selection early; it is **not** ground-truth arithmetic labels.

- Weight constant: `kExecSupervisionWeight = 0.05f` in `AutogradTraining.cu` (added to total loss with numeric loss).

---

## Configuration (`LanguageModelConfig`)

Fields in `grim_language_model_cuda.hpp` (defaults shown where typical):

| Field | Role |
|--------|------|
| `execution_block_enabled` | Master switch. |
| `execution_block_layer` | Encoder index **after which** the block runs. **`-1`** means `num_layers - 2` (clamped into `[0, num_layers-1]`). |
| `execution_block_num_ops` | Usually **8** (Add, Sub, Mul, Div, Mod, Pow, Min, Max). |
| `execution_block_num_slots` | **V** — number of memory slots (`ExecutionMemory`). |
| `execution_block_num_steps` | **K** — execution steps per forward at `execution_block_layer`. |
| `execution_block_d_key` | Key dimension for memory addressing / cross-attn keys. |
| `execution_block_d_type` | Per-slot type embedding width (numeric path). |
| `execution_block_cross_attn_head_dim` | Head dim for the read cross-attention. |
| `execution_block_cross_attn_topk` | Top-k keys per query (default **1**). |
| `execution_block_usage_decay` | Decay for slot usage statistics after reads. |
| `execution_block_diversity_kappa` | Penalty for rewriting the same slot repeatedly. |
| `execution_block_memory_slot_bias` | Bias against picking memory slots as args (balances vs. atom decode). |

`ExecutionBlockConfig` in `execution_block_GPU.hpp` also carries decode MLP sizes, `empty_slot_bonus`, stream/handle, etc. Training construction (`TrainingOps.cu` / `InitinferenceState.cu`) maps the `LanguageModel` fields above into `ExecutionBlockConfig`; constants like `empty_slot_bonus` use struct defaults unless you extend the LM config.

---

## Runtime placement (autograd forward)

Order inside the encoder loop (`AutogradTraining.cu`):

1. Run encoder layer `layer_idx` → `layer_output`.
2. If **`layer_idx == exec_layer`**: allocate/clear `intermediates.exec_memory`, then for **`step = 0 … K-1`** call `executeStep(...)` and append each `ExecutionBlockStepOutput` to `intermediates.execution_block_output.steps`.
3. If **`layer_idx >= exec_layer`** and **`exec_memory.num_filled > 0`**: call **`crossAttentionRead(layer_output, exec_memory, ...)`** (mutates `layer_output` in place).
4. Push `layer_output` into `encoder_layer_outputs`.

So: **write** happens once at `exec_layer` (K steps); **read** happens at that layer and every deeper layer until the stack ends.

---

## Public API — `ExecutionBlockLayer`

Declared in `execution_block_GPU.hpp`, implemented in `execution_block_GPU.cu`.

### `executeStep(...)`

**Purpose:** One register-machine step: build candidates, STE-select args and op, compute scalar, STE-select write slot, update `M`.

**Inputs:**

| Argument | Shape / type | Notes |
|----------|----------------|-------|
| `hidden_states` | `Tensor` `[total_tokens, d_model]` | Encoder state at `exec_layer` (read-only for this step’s logic). |
| `atom_embeddings` | `const float*` `[num_atoms, atom_embedding_dim]` | From ScratchBlock device buffer. |
| `atom_positions` | `const int*` `[num_atoms]` | Token indices of atoms. |
| `num_atoms` | `int` | May be 0; pool can still include memory. |
| `total_tokens` | `int` | Sequence length (batch collapsed as elsewhere). |
| `M` | `ExecutionMemory&` | Must be allocated and cleared for this forward; updated in place. |
| `step` | `int` | `0 … K-1` for step embedding and bookkeeping. |
| `stream` | `cudaStream_t` | |

**Output:** `ExecutionBlockStepOutput` — selected indices, decoded/computed floats (host-side scalars where filled), and **tensors** `op_logits`, `arg1_scores`, `arg2_scores`, `write_logits` (used for logging, supervision, and autograd as wired).

**Validation:** `validateExecuteStepInputsOrThrow` (and internal checks) — invalid shapes/pointers fail loudly.

### `crossAttentionRead(...)`

**Purpose:** Let every token hidden state attend to **valid** memory slots (sharpened, top-k, gated), and **decay-update** `M.usage`.

**Inputs:**

| Argument | Notes |
|----------|--------|
| `hidden_states` | `Tensor&` `[total_tokens, d_model]` — **updated in place** (`H += g * W_O(R)`). |
| `M` | `ExecutionMemory&` — `usage` updated; read uses `key_embeds` / `state_embeds` / `valid_mask` per implementation. |
| `total_tokens` | `int` |
| `stream` | `cudaStream_t` |

**Validation:** `validateCrossAttentionInputsOrThrow`.

### Other helpers

- `validateConfigOrThrow()`, `validateMemoryOrThrow(M)`
- `setStream` / `setCublasHandle` — must be consistent with training stream/handle.
- **Weights:** accessors `w_decode_*`, `W_op_select`, `W_state`, `W_key_base`, write-head tensors, read-head tensors, `tau`, etc. — registered under `ParamGroupType::EXECUTION_BLOCK` for optimizers and grad norms.

---

## `ExecutionMemory` (buffers)

Allocated via `ExecutionMemory::allocate(V, atom_dim, d_model, d_key, d_type, stream)` and cleared with `clear(stream)`.

| Member | Typical shape | Role |
|--------|----------------|------|
| `values` | `[V, 1]` | Scalar stored per slot. |
| `atom_embeds` | `[V, atom_dim]` | ScratchBlock-style encoding of result. |
| `state_embeds` | `[V, d_model]` | Content side for cross-attn **V** (and related paths). |
| `valid_mask` | `[V]` | Slot occupied. |
| `usage` | `[V]` | Decayed usage from reads. |
| `write_score` | `[V]` | Learned overwrite bias (not zeroed on `clear` by design). |
| `key_embeds` | `[V, d_key]` | Addressing keys (from result embedding path, not copied from `state_embeds`). |
| `type_embed` | `[V, d_type]` | Type tag (numeric). |
| `recent_write_mask` | `[V]` | Diversity / last-step write tracking. |
| `num_filled` | `int` | Bookkeeping for read gating path. |

---

## Training integration — where to look in code

| Concern | Location |
|---------|-----------|
| Encoder loop, `executeStep` / `crossAttentionRead` | `training/Autograd/AutogradTraining.cu` |
| Intermediates ownership | `training/Autograd/AutogradIntermediates.hpp` — `exec_memory`, `execution_block_output` |
| `AutogradContext` pointer | `training/Autograd/AutogradTraining.hpp` — `execution_block` |
| Context init | `initAutogradContext(..., execution_block, ...)` — call sites pass `model.getExecutionBlockLayer()` |
| Build layer | `training/TrainingOps.cu`, `Layers/InitInferenceState/InitinferenceState.cu` |
| Optimizer registration | `training/LanguageModel_Training.cu` — all Execution Block tensors |
| Gradient norm bucket | `Shared/GradNorm/GradNormGPU.*`, `Phase2_TrainingLoop.cu` — `execution_block_*` |

---

## Checkpoints / FlatBuffers

Execution Block weights are stored in the main transformer FlatBuffer as **`ExecutionBlockWeights`** inside **`TransformerModel.execution_block`** (`grim_transformer_model_generated.h`).

- **Save/load path:** `Layers/Serialization/Serialization_GPU.cu` packs/unpacks the 21 weight vectors; `Common/grim_model_serialization.cu` fills `SerializationSaveSources.execution_block` / `SerializationLoadRequest.execution_block` from `LanguageModel::getExecutionBlockLayer()`.

Older checkpoints without `execution_block` still load; missing tables leave initialized weights.

---

## Tensor shapes in host code

`Tensor::shape` is a `TensorContract::TensorShape`. For **2D** (BSM) tensors, use:

- `tensor.shape.flat.rows`
- `tensor.shape.flat.cols`

Do not use `tensor.shape.rows` / `.cols` (they do not exist).

---

## Reasoning Head interaction

- **Reasoning Head** still produces `ReasoningHeadOutput` (op / arg1 / arg2 logits) for explainability and, when enabled, **pseudo-targets** for Execution Block supervision.
- Candidate indices for args span **atoms + memory slots** in the Execution Block; Reasoning Head arg logits are sized for **`num_atoms`** only — supervision aligns where dimensions match; edge cases (e.g. zero atoms) are handled with guards in loss code.

---

## Quick checklist for a new experiment

1. Enable ScratchBlock + Execution Block in config.  
2. Set `execution_block_layer` (or rely on `-1` default).  
3. Tune `execution_block_num_slots` (V) and `execution_block_num_steps` (K).  
4. Optionally enable Reasoning Head for the 0.05-weight supervision term.  
5. Verify checkpoint save/load includes `execution_block` in the FlatBuffer (new saves).  
6. Use `getExecutionBlockLayer()` only when `execution_block_enabled` is true; otherwise `nullptr`.

For kernel-level behavior and STE details, see comments and validation macros in `execution_block_GPU.cu`.
