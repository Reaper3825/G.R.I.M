# GRIM-text parameter group registry

Source-reviewed: 2026-10-15. This describes the current code paths, not a dump of a running model or a particular configuration. Registration is compiled under `USE_CUDA`.

## Reading the tables

Each `ParameterGroup` represents one tensor, not an entire layer. **Feature level** means architectural scope. **Owner** means the durable C++ tensor owner, not the optimizer category. **Activation** describes the forward operation; the separate condition column describes when the group is registered.

Notation: `D = d_model`, `F = d_ff`, `M = lm_head_mlp_d_ff`, `V = vocab_size`, `Q = qkv_dim`, `H = num_heads`, `T = Tokenizer::kAtomTypeCount`. Layer `i` is zero-based. Shapes are stored tensor shapes. Configuration names refer to resolved training configuration fields unless explicitly described as construction HP fields.

The base-group tables apply to ordinary training (`lora_model=false`). In LoRA mode, those same registration helpers populate the base checkpoint inventory, while only enabled LoRA A/B tensors enter the optimizer inventory. Frozen RMS gains are omitted from both inventories by these helpers. Registration does not guarantee a nonzero gradient on every batch: the feature must participate in the forward/loss path.

## Ownership and lifecycle

| Object / stage | Owner or entry point | Responsibility |
| --- | --- | --- |
| Durable parameter storage | `TrainingContext::parameter_registry` / `StartupParameterRegistry` | Owns embedding, LM-head, encoder, FFN, optional feature bundles, and LoRA pairs. |
| Model topology | `GpuModelState` / `assembleGpuModel()` | Constructs compute layers that borrow registry tensors; sets `base_requires_grad = !lora_model`. |
| Trainable inventory | `StartupParameterRegistry::parameter_groups` | Owns group metadata; each group borrows its parameter tensor. |
| Base checkpoint inventory | `StartupParameterRegistry::checkpoint_parameter_groups` | Separate non-owning tensor inventory. In ordinary training it copies the registered base groups; in LoRA mode it is built separately with gradients not required. |
| Registration | `buildParameterGroups()` | Discovers configured tensors, validates names/storage/precision, and swaps in the validated trainable inventory. |
| Optimizer moments | `OptimizerState::m_states`, `v_states` | Owns first/second moment tensors. `bindOptimizerState()` allocates by group size and attaches borrowed pointers. |
| Forward activations | Forward-output structures and compute paths | Intermediate values, not `ParameterGroup` entries. |

Owner abbreviations below are members of `StartupParameterRegistry`:

| Alias | Registry member | C++ tensor bundle |
| --- | --- | --- |
| Embedding | `embedding_parameters` | `EmbeddingParameterTensors` |
| Head | `lm_head_parameters` | `LMHeadParameterTensors` |
| Encoder[i] | `encoding_layer_parameter_tensors[i]` | `EncodingLayerParameterTensors` |
| FFN[i] | `feed_forward_parameter_tensors[i]` | `FeedForwardParameterTensors` |
| Atom boundary | `atom_insertion_boundary_parameters` | `AtomInsertionBoundaryParameterTensors` |
| Retrieval | `local_atom_retrieval_parameters` | `LocalAtomRetrievalParameterTensors` |
| LoRA[i,class] | `lora_layer_parameter_pairs[i].pairs[class index]` | `LoRAParameterPair` |

## Global input and output groups

Evidence: [tensor owners and specifications](ParameterRegistry.hpp), [registerTopLevelParameters and initialization](ParameterGroupRegistration.cu), [LM-head forward](../../../../Layers/LMHead/lm_head_GPU.cu).

| Exact group name | Feature level / what it controls | Owner.member; shape | Registration condition | Type / stats bucket | Activation / forward role |
| --- | --- | --- | --- | --- | --- |
| `embedding` | Global input; token representations | Embedding.`token_weights`; `[V,D]` | `!tie_embeddings` | `EMBEDDING / EMBEDDING` | Token lookup; no learned nonlinear activation. |
| `lm_head_weight` | Global output; vocabulary readout | Head.`weights`; `[V,D]` | `!tie_embeddings` | `LM_HEAD / LM_HEAD` | Linear vocabulary logits from head hidden states. |
| `embedding_lm_head_tied` | Shared input/output vocabulary matrix | Head.`weights` aliases Embedding.`token_weights`; `[V,D]` | `tie_embeddings` | `LM_HEAD / EMBEDDING` | One storage/gradient allocation used for lookup and output projection; registered once. |
| `lm_head_bias` | Global output; per-token logit offset | Head.`bias`; `[V]` | `lm_head_bias_enabled` | `LM_HEAD / LM_HEAD` | Additive bias; optional output-unigram prior initializes this same tensor. |
| `final_rms_gamma` | Global output; pre-head normalization gain | Head.`final_rms_gamma`; `[D]` | `!freeze_learned_rms_gammas` | `RMSNORM / LM_HEAD` | RMSNorm gain; normalization still exists when gain is frozen. |
| `lm_head_mlp_w_gate` | Global output adapter; gate projection | Head.`mlp_W_gate`; `[D,M]` | `lm_head_mlp_enabled` | `LM_HEAD / LM_HEAD` | `SiLU(z @ W_gate)` where `z` is normalized encoder output. |
| `lm_head_mlp_w_up` | Global output adapter; content projection | Head.`mlp_W_up`; `[D,M]` | `lm_head_mlp_enabled` | `LM_HEAD / LM_HEAD` | Linear `z @ W_up`, multiplied by the SiLU gate. |
| `lm_head_mlp_w_down` | Global output adapter; return to model width | Head.`mlp_W_down`; `[M,D]` | `lm_head_mlp_enabled` | `LM_HEAD / LM_HEAD` | Linear down projection and residual addition: `u = z + lm_head_mlp_alpha * (SiLU(z @ W_gate) * (z @ W_up)) @ W_down`. |

`lm_head_mlp_alpha` is a configuration scalar, not a parameter group. Tied embedding/head precision settings must match, even though the shared group's type is `LM_HEAD`. Readout math is row-local; historical residual/hidden/weight/logit centering and PC1 projection are not part of this registry.

## Per-encoder-layer groups

All names expand `i` to the layer number, e.g. `layer0_qkv_weight`. All have `layer_index=i` and stats bucket `ENCODER`. Evidence: [registerEncodingLayerParameters / registerFeedForwardParameters](ParameterRegistry.hpp), [initializers and registration](ParameterGroupRegistration.cu), [encoder forward](../../../../Layers/Encoding/Encoding_GPU.cu), [attention head gates](../../../../Layers/FlashAttention/EncoderSelfAttention_GPU.cu), [FFN forward](../../../../Layers/FeedForward/Feed_Forward_GPU.cu).

| Exact name pattern | Feature level / what it controls | Owner.member; shape | Registration condition | Type | Activation / forward role |
| --- | --- | --- | --- | --- | --- |
| `layer{i}_qkv_weight` | Layer attention; query/key/value projections | Encoder[i].`W_qkv`; `[Q,D]` | Every configured layer | `ATTENTION` | Linear Q/K/V projection feeding attention. |
| `layer{i}_qkv_bias` | Layer attention; Q/K/V offsets | Encoder[i].`b_qkv`; `[Q]` | `attention_qkv_bias_enabled` | `ATTENTION` | Additive projection bias. |
| `layer{i}_wo_weight` | Layer attention; output mixing | Encoder[i].`W_o`; `[D,D]` | Every configured layer | `ATTENTION` | Linear attention output projection. |
| `layer{i}_wo_bias` | Layer attention; output offset | Encoder[i].`b_o`; `[D]` | `attention_output_bias_enabled` | `ATTENTION` | Additive output bias. |
| `layer{i}_rms1_gamma` | Layer normalization before attention | Encoder[i].`rms1_gamma`; `[D]` | `!freeze_learned_rms_gammas` | `RMSNORM` | Learned RMSNorm gain. |
| `layer{i}_rms2_gamma` | Layer normalization before FFN | Encoder[i].`rms2_gamma`; `[D]` | `!freeze_learned_rms_gammas` | `RMSNORM` | Learned RMSNorm gain. |
| `layer{i}_layer_scale1` | Layer residual; attention branch amplitude | Encoder[i].`layer_scale1`; `[1,D]` | `use_layer_scale` | `RMSNORM` | Per-channel multiplication before attention residual addition. |
| `layer{i}_layer_scale2` | Layer residual; FFN branch amplitude | Encoder[i].`layer_scale2`; `[1,D]` | `use_layer_scale` | `RMSNORM` | Per-channel multiplication before FFN residual addition. |
| `layer{i}_attention_residual_gate_weight` | Layer attention; token-wise residual gate | Encoder[i].`attention_residual_gate.W_gate`; `[D,1]` | `attention_residual_gate_enabled` | `ATTENTION` | Normalized input projects to one logit per token; multiplier is `2 * sigmoid(logit)`. |
| `layer{i}_attention_residual_gate_bias` | Layer attention; residual gate offset | Encoder[i].`attention_residual_gate.b_gate`; `[1]` | `attention_residual_gate_enabled` | `ATTENTION` | Added before residual-gate sigmoid; scales the projected attention branch. |
| `layer{i}_attention_head_gate_weight` | Layer attention; token/head-wise gating | Encoder[i].`attention_head_gate.W_gate`; `[D,H]` | `attention_head_gate_enabled` | `ATTENTION` | Normalized input projects to H logits; `2 * sigmoid(logits)` scales heads before output projection. |
| `layer{i}_attention_head_gate_bias` | Layer attention; per-head gate offsets | Encoder[i].`attention_head_gate.b_gate`; `[H]` | `attention_head_gate_enabled` | `ATTENTION` | Added before head-gate sigmoid. |
| `layer{i}_ffn_w_gate` | Layer FFN; SwiGLU gate | FFN[i].`W_gate`; `[D,F]` | Every configured layer | `FFN` | `SiLU(x @ W_gate)`. |
| `layer{i}_ffn_w1` | Layer FFN; up/content projection | FFN[i].`W1`; `[D,F]` | Every configured layer | `FFN` | Linear `x @ W1`, multiplied by the SiLU gate. |
| `layer{i}_ffn_w2` | Layer FFN; down projection | FFN[i].`W2`; `[F,D]` | Every configured layer | `FFN` | Projects gated content back to D dimensions. |
| `layer{i}_ffn_b2` | Layer FFN; output offset | FFN[i].`b2`; `[1,D]` | `ffn_output_bias_enabled` | `FFN` | Additive bias after down projection; no FFN gate/up bias groups. |

The independent bias flags above are what registration actually uses; older bundle comments referring generically to `use_bias` should not be read as the current registration rule. LayerScale uses the `RMSNORM` category but is multiplication, not RMS normalization. Freezing RMS gains does not disable LayerScale registration.

## Optional global feature groups

All have `layer_index=-1`. Evidence: [optional registration helpers and buildParameterGroups](ParameterGroupRegistration.cu), [assembly call sites](ModelGpuAssembly.cu), [boundary projection](../../../../Shared/AtomInsertion/AtomInsertionBoundaryProjection.cu), [retrieval forward](../../../../Shared/LocalAtomRetrieval/LocalAtomRetrievalForward.cu).

| Exact group name | Feature level / what it controls | Owner.member; shape | Registration condition / wiring | Type / stats bucket | Activation / forward role |
| --- | --- | --- | --- | --- | --- |
| `atom_insertion_left_projection_weight` | Global auxiliary feature; left boundary context | Atom boundary.`left_projection_weight`; `[D,D]` | `atom_insertion_enabled` | `LM_HEAD / LM_HEAD` | Linear projection of left adjacent state. |
| `atom_insertion_right_projection_weight` | Global auxiliary feature; right boundary context | Atom boundary.`right_projection_weight`; `[D,D]` | `atom_insertion_enabled` | `LM_HEAD / LM_HEAD` | Linear projection of right adjacent state, added to left projection. |
| `atom_insertion_projection_bias` | Global auxiliary feature; boundary representation offset | Atom boundary.`projection_bias`; `[1,D]` | `atom_insertion_enabled` | `LM_HEAD / LM_HEAD` | Bias added to summed boundary projections; no separate nonlinear activation in the boundary projection primitive. |
| `local_atom_retrieval_type_no_reference_key` | Global retrieval; learned no-reference candidate per atom type | Retrieval.`type_no_reference_key`; `[T,D]` | `local_atom_retrieval_enabled` | `ARG_SELECTOR / ENCODER` | Candidate key used in retrieval scoring when choosing no reference; this tensor is not itself an activation function. |

Enabling a feature is distinct from having usable examples for its loss on a given batch. The registry does not encode batch-level target availability. Disabled optional atom/retrieval owners are rejected if they still exist. Disabled config-gated tensors are rejected if they retain data or gradient storage.

## LoRA groups

Evidence: `makeLoRAMatrixSpecs()`, `initializeLoRAParameterTensors()`, `registerLoRAParameters()`, and `buildParameterGroups()` in [ParameterGroupRegistration.cu](ParameterGroupRegistration.cu).

Requires `lora_model=true` plus the corresponding `LoRATrainingHP` class setting `.enabled`. Each enabled class creates **two groups per layer**. All are owned by LoRA[i,class], use stats bucket `ENCODER`, and have `layer_index=i`. `r` below is that class's rank; precision comes from that class's `.precision`, not the base type's precision field.

| Exact group names | Feature / class setting | A shape; B shape | Type | Forward effect / activation |
| --- | --- | --- | --- | --- |
| `layer{i}_qkv.lora_A`, `layer{i}_qkv.lora_B` | QKV adaptation; `qkv` | `[r,D]`; `[Q,r]` | `ATTENTION` | Low-rank update to QKV projection; no new nonlinear activation. |
| `layer{i}_wo.lora_A`, `layer{i}_wo.lora_B` | Attention output adaptation; `o` | `[r,D]`; `[D,r]` | `ATTENTION` | Low-rank update to attention output projection. |
| `layer{i}_ffn_w_gate.lora_A`, `layer{i}_ffn_w_gate.lora_B` | SwiGLU gate adaptation; `gate` | `[r,F]`; `[D,r]` | `FFN` | Low-rank update to gate projection, before existing SiLU. |
| `layer{i}_ffn_w1.lora_A`, `layer{i}_ffn_w1.lora_B` | FFN up adaptation; `w1` | `[r,F]`; `[D,r]` | `FFN` | Low-rank update to up projection. |
| `layer{i}_ffn_w2.lora_A`, `layer{i}_ffn_w2.lora_B` | FFN down adaptation; `w2` | `[r,D]`; `[F,r]` | `FFN` | Low-rank update to down projection. |

The stored matrix update is scaled by `alpha / rank`. Registration validates class, rank, alpha, precision, shape, and finite positive scale against authored settings. All base parameters must be frozen, including optional feature parameters. LoRA groups receive weight-decay multiplier `0.0` and learning-rate multiplier `1.0`. Internal tensor allocation labels such as `lora_qkv_A` are not the registered names shown above.

## Optimizer metadata and conditions

Evidence: `Registrar` in [ParameterGroupRegistration.cu](ParameterGroupRegistration.cu) and [ParameterGroup definition](../../../../Shared/TensorContract/TensorContract_GPU.hpp).

| Field / policy | Current registration behavior | Meaning |
| --- | --- | --- |
| `name` | Unique, nonempty name | Stable identity within this inventory. |
| `tensor` | Borrowed pointer; nonnull data and nonzero size required | Source of weights, gradients, and element count. Trainable registration additionally requires a gradient buffer. |
| `type` | One of six categories below | Optimizer/analysis category and default precision lookup. |
| `stats_bucket` | `EMBEDDING`, `ENCODER`, or `LM_HEAD`; sentinel rejected | Parameter accounting, independent of `type`. |
| `parameter_precision` | Only `FP32` accepted here | `BF16_COMPUTE` is explicitly rejected because implicit BF16 consumers are not wired for these optimizer-visible tensors; unspecified precision is also rejected. |
| `layer_index` | `-1` globally; `i` for encoder/LoRA groups | Depth identity. |
| `weight_decay_multiplier` | Base groups `1.0`; LoRA `0.0` | Registration multiplier, not the full effective optimizer weight decay. No special zero-decay exception is stamped here for base biases or RMS gains. |
| `lr_multiplier` | `1.0` | Registration multiplier, not the full effective learning rate. |
| `upsilon` | Defaults to `1.0`; with `use_depth_aware_upsilon` and `i >= 0`, `UPSILON_BASE * sqrt(UPSILON_REFERENCE_LAYERS / (i + 1))` | Depth-aware regularization metadata. Global groups retain the default. |
| `m_tensor`, `v_tensor` | Cleared during registration; attached during optimizer binding | Borrowed moments; checkpoint inventory keeps these unbound. |
| `gradient_verification` | Initializes observed delivery count from the tensor | Tracks gradient delivery / signal verification history. |

| Group type | Base precision configuration field | Included features |
| --- | --- | --- |
| `EMBEDDING` | `parameter_precision_embedding` | Untied input embedding. |
| `LM_HEAD` | `parameter_precision_lm_head` | Output weight/bias, tied vocabulary matrix, head MLP, atom boundary projections. |
| `ATTENTION` | `parameter_precision_attention` | QKV/output projections, biases, residual/head gates. |
| `FFN` | `parameter_precision_ffn` | SwiGLU gate/up/down and output bias. |
| `RMSNORM` | `parameter_precision_rmsnorm` | Encoder/final RMS gains and LayerScale. |
| `ARG_SELECTOR` | `parameter_precision_arg_selector` | Local retrieval no-reference keys. |

## Initialization and validation notes

| Feature | Startup initialization / invariant |
| --- | --- |
| Encoder QKV, FFN gate/up | Xavier uniform. |
| Encoder output and FFN down | Xavier uniform with `residual_projection_init_gain`. |
| Encoder RMS gains | Ones; gradient allocation requires base training and unfrozen RMS gains. |
| LayerScale | Filled with `layer_scale_init`. |
| Attention residual/head gates | Weights and biases start at zero, so `2 * sigmoid(0) = 1`; both initially preserve branch/head amplitude. |
| Attention and FFN biases | Zero when their individual feature flags enable allocation. |
| Tied vocabulary weights | Embedding/head must alias when tied and must not alias when untied; mismatched precision is rejected. |
| Duplicate storage | Duplicate data pointers are rejected to prevent stepping the same memory twice. |
| Topology | Encoder exists, every configured layer exists, and encoder/FFN owner vectors match `num_layers`. |
| Optimizer binding | Requires a valid CUDA stream; moment allocation and each group's sizes are validated. |

## Inventory counting aid

For ordinary training, let `L=num_layers`; each boolean below contributes 0 or 1. This counts registered tensors, not scalar parameters:

```text
global = (tie_embeddings ? 1 : 2)
       + lm_head_bias_enabled
       + 3 * lm_head_mlp_enabled
       + !freeze_learned_rms_gammas
       + 3 * atom_insertion_enabled
       + local_atom_retrieval_enabled

per_layer = 5                         # QKV, output, FFN gate/up/down
          + attention_qkv_bias_enabled
          + attention_output_bias_enabled
          + ffn_output_bias_enabled
          + 2 * !freeze_learned_rms_gammas
          + 2 * use_layer_scale
          + 2 * attention_residual_gate_enabled
          + 2 * attention_head_gate_enabled

base_group_count = global + L * per_layer
lora_trainable_group_count = 2 * L * enabled_LoRA_class_count
```

Configuration validation and tensor/owner invariants must also pass; these formulas do not imply every combination is supported. Scalar parameter counts are the products of the shapes in the tables, summed only over included groups, with the tied vocabulary matrix counted once.
