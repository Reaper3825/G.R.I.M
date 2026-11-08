# GRIM-text Learned Parameters Reference

This inventory describes the current source, not a fixed model-size example.
Tensor owners and inventory slices are defined in
[ParameterRegistry.hpp](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterRegistry.hpp);
[ParameterGroupRegistration.cu](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.cu)
allocates, validates, and registers the configured tensors.

## Dimensions and authority

Use the finalized Phase 1 config snapshot and the actual tensor shapes:

| Symbol | Meaning |
|---|---|
| `D` | `d_model` |
| `F` | Encoder `d_ff` |
| `M` | `lm_head_mlp_d_ff` |
| `H`, `Hkv` | Query and KV head counts |
| `d` | `D / H` |
| `G` | Fused QKV width: `D + 2 * Hkv * d` |
| `V` | Actual vocabulary size, not execution slots |
| `A` | Sequence-local retrieval atom-type count |
| `r` | Rank for the particular enabled LoRA matrix class |

[HyperparameterGroupings.hpp](../../resources/models/GRIM-text/Shared/HyperParameters/HyperparameterGroupings.hpp)
derives grouped read views from the authoritative config. Layer count, bias
gates, precision, freezing, and feature gates are model/config facts, not
constants maintained by this document.

## Base parameter inventory

### Embedding and LM head

| Owner field | Registered name | Stored shape | Gate / policy |
|---|---|---|---|
| `EmbeddingParameterTensors::token_weights` | `embedding` | `[V,D]` | Registered separately only when untied |
| `LMHeadParameterTensors::weights` | `lm_head_weight` | `[V,D]` | Untied head |
| Shared embedding/head weight | `embedding_lm_head_tied` | `[V,D]` | One entry when `tie_embeddings=true` |
| `bias` | `lm_head_bias` | `[V]` | `lm_head_bias_enabled` |
| `final_rms_gamma` | `final_rms_gamma` | `[D]` | Excluded from trainable inventory when `freeze_learned_rms_gammas=true` |
| `mlp_W_gate` | `lm_head_mlp_w_gate` | `[D,M]` | `lm_head_mlp_enabled` |
| `mlp_W_up` | `lm_head_mlp_w_up` | `[D,M]` | `lm_head_mlp_enabled` |
| `mlp_W_down` | `lm_head_mlp_w_down` | `[M,D]` | `lm_head_mlp_enabled` |

The embedding owner contains only token weights. There is no current learned
position-weight or embedding RMS-gamma inventory. Positional behavior is
implemented through the configured positional-bias/rotation path.

The head-side SwiGLU residual adapter is distinct from LoRA:
`z = RMSNorm(h)`, followed, when enabled, by
`u = z + alpha * ((SiLU(z @ W_gate) * (z @ W_up)) @ W_down)`.
`alpha` is an authored scalar, not an additional learned tensor. See
[lm_head_GPU.cu](../../resources/models/GRIM-text/Layers/LMHead/lm_head_GPU.cu).

### Encoder attention and normalization

Names below use `layer{l}_` for zero-based layer `l`.

| Owner field | Registered suffix | Stored shape | Gate / policy |
|---|---|---|---|
| `W_qkv` | `qkv_weight` | `[G,D]` | Required fused GQA projection |
| `b_qkv` | `qkv_bias` | `[G]` | `attention_qkv_bias_enabled` |
| `W_o` | `wo_weight` | `[D,D]` | Required output projection |
| `b_o` | `wo_bias` | `[D]` | `attention_output_bias_enabled` |
| `rms1_gamma` | `rms1_gamma` | `[D]` | Trainable unless RMS gammas are frozen |
| `rms2_gamma` | `rms2_gamma` | `[D]` | Trainable unless RMS gammas are frozen |
| `layer_scale1` | `layer_scale1` | `[1,D]` | `use_layer_scale` |
| `layer_scale2` | `layer_scale2` | `[1,D]` | `use_layer_scale` |
| Residual gate `W_gate` | `attention_residual_gate_weight` | `[D,1]` | `attention_residual_gate_enabled` |
| Residual gate `b_gate` | `attention_residual_gate_bias` | `[1]` | Same gate |
| Head gate `W_gate` | `attention_head_gate_weight` | `[D,H]` | `attention_head_gate_enabled` |
| Head gate `b_gate` | `attention_head_gate_bias` | `[H]` | Same gate |

LayerScale is a feature vector, not a single scalar. There are no registered
per-head `alpha_q`/`alpha_k` tensors or sandwich-normalization gains.

### Encoder feed-forward

These are the registry's stored shapes; do not transpose the inventory merely
to match a preferred mathematical notation.

| Owner field | Registered suffix | Stored shape | Gate |
|---|---|---|---|
| `W_gate` | `ffn_w_gate` | `[D,F]` | Required |
| `W1` | `ffn_w1` | `[D,F]` | Required |
| `W2` | `ffn_w2` | `[F,D]` | Required |
| `b2` | `ffn_b2` | `[1,D]` | `ffn_output_bias_enabled` |

### Atom insertion and sequence-local retrieval

| Owner field | Registered name | Stored shape | Gate |
|---|---|---|---|
| `left_projection_weight` | `atom_insertion_left_projection_weight` | `[D,D]` | `atom_insertion_enabled` |
| `right_projection_weight` | `atom_insertion_right_projection_weight` | `[D,D]` | Same gate |
| `projection_bias` | `atom_insertion_projection_bias` | `[1,D]` | Same gate |
| `type_no_reference_key` | `local_atom_retrieval_type_no_reference_key` | `[A,D]` | `local_atom_retrieval_enabled` |

These owners are `AtomInsertionBoundaryParameterTensors` and
`LocalAtomRetrievalParameterTensors`. Atom insertion is grouped as LM-head
parameters; retrieval uses `ARG_SELECTOR` precision/type metadata and encoder
statistics. That enum name does not imply the removed decode-time slot selector
is active.

## LoRA inventory

When `lora_model=true`, `parameter_groups` contains only the enabled trainable
A/B pairs. The base tensors must be frozen. `LoRALayerParameterPairs` supports
five matrix classes, each with its own rank/alpha/precision:

| Matrix class | Registered names, prefix `layer{l}.` | A shape | B shape |
|---|---|---|---|
| QKV | `qkv.lora_A`, `qkv.lora_B` | `[r,D]` | `[G,r]` |
| Attention output | `wo.lora_A`, `wo.lora_B` | `[r,D]` | `[D,r]` |
| FFN gate | `ffn_w_gate.lora_A`, `ffn_w_gate.lora_B` | `[r,F]` | `[D,r]` |
| FFN up | `ffn_w1.lora_A`, `ffn_w1.lora_B` | `[r,F]` | `[D,r]` |
| FFN down | `ffn_w2.lora_A`, `ffn_w2.lora_B` | `[r,D]` | `[F,r]` |

These shapes come from `makeLoRAMatrixSpecs()` and `registerLoRAParameters()`.
The separate `checkpoint_parameter_groups` records the base inventory.
Do not include frozen base tensors in the LoRA optimizer or mistake
model-side adapter support for completion of the planned MMO personalization
queue. See [GRIM_TEXT_LORA_SUPPORT_PLAN.md](plans/LORA_PERSONALIZATION.md).

## Registration, tying, and optimizer policy

- Tied embedding/head data and gradients alias, and are registered exactly once.
  Matmul and embedding contributions accumulate into the shared buffer.
- `ParameterGroup` borrows a tensor and optimizer moment bindings. Duplicate
  names or duplicate data-pointer registration reject the complete transaction.
- Base registrations use `wd_mult=1.0` and `lr_mult=1.0` unless an explicit
  registration changes them. Final gamma does not currently receive the old
  hardcoded `lr_mult=0.1`.
- LoRA registration uses `wd_mult=0.0`, `lr_mult=1.0`, and class-specific precision.
- Optimizer step implementations apply `effective_lr = learning_rate *
  group.lr_multiplier` and effective decay using `group.upsilon` and
  `group.weight_decay_multiplier`.
- Precision metadata selects the compute policy; current stored parameter and
  checkpoint payload bytes are FP32.

Optimizer behavior is implemented in
[AdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu)
and [RAdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.cu).

## Checkpoint format

[ParameterCheckpoint.cu](../../resources/models/GRIM-text/Common/ParameterCheckpoint.cu)
and [grim_parameter_checkpoint.fbs](../../resources/models/GRIM-text/training/schemas/grim_parameter_checkpoint.fbs)
use identifier `GRCP`, extension `.grimckpt`, generic `ParameterEntry` records,
exact ordered manifests, semantic facts, per-entry/payload checksums, and
post-write verification. The schema does not contain active architecture-specific
weight tables.

Base checkpoint I/O and LoRA training resume I/O have different inventories.
[LoRATrainingCheckpoint.hpp](../../resources/models/GRIM-text/Common/LoRATrainingCheckpoint.hpp)
defines the separate LoRA training format; registry enum values reserved for
retired components are not current tensor owners.

ReasoningHead, neural ExecutionBlock/register-machine parameters, NumberEncoder,
SlotSeedEncoder, decode-time slot selector, learned uncertainty-loss weights,
and conventional MTP heads are not part of the current live inventory.
Historical architecture documents are under [legacy](../legacy).
