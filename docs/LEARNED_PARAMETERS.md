# GRIM-text Learned Parameters Reference

> Extracted from the serialization layer (`Serialization_views.hpp`, `Serialization_save.cu`, `grim_transformer_model.fbs`).
> All shapes assume runtime config from `ai_config.json`: `d_model=768`, `num_heads=12`, `num_kv_heads=4`, `head_dim=64`, `d_ff=3072`, `num_layers=6`, `max_seq_len=2048`, `V=8` (execution slots), `K=4` (execution steps), `num_ops=4`, `d_key=64`, `d_type=16`, `atom_embedding_dim=64`.

---

## Architecture Overview

```
Input tokens
    │
    ▼
┌──────────────────┐
│  Embedding Layer  │ ← token_weights [V×768], rms_gamma [768]
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  ScratchBlock     │ ← atom_type_embeddings, atom_projection
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Encoder Layer ×6  │ ← W_qkv, W_o, W_gate, W1, W2, RMSNorm γ, LayerScale
└──────────────────┘
    │
    ├─────────────────────────┐
    ▼                         ▼
┌──────────────┐   ┌──────────────────┐
│ final_rms_γ  │   │ ExecutionBlock   │ ← 29 parameter tensors
└──────────────┘   └──────────────────┘
    │                         │
    ▼                         ▼
┌──────────────┐   ┌──────────────────┐
│   LM Head    │   │  ReasoningHead   │ ← W_op, b_op, W_arg1, W_arg2
└──────────────┘   └──────────────────┘
```

---

## 1. Embedding Layer

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `token_weights` | [vocab_size, 768] | `embeddings.token_embeddings` | **TIED** with LM head when `tie_embeddings=true` (default). Single GPU buffer — never zero or free both. |
| `position_weights` | [2048, 768] | `embeddings.positional_encodings` | **Conditional** — only when `positional_encoding == NONE` (learned mode). Not allocated for ALiBi/RoPE. |
| `rms_gamma` | [768] | `embeddings.rms_gamma` | Post-embedding RMSNorm scale. Conditional on `use_rms_norm`. |

---

## 2. Encoder Layer (×`num_layers`)

Each of the 6 encoder layers contains:

### Attention Sub-layer

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_qkv` | [1280, 768] | `attention.w_qkv_data` | Fused Q+K+V. Rows = `num_heads×head_dim + 2×num_kv_heads×head_dim` = 768+512 = 1280. |
| `b_qkv` | [1280] | `attention.b_qkv_data` | Conditional on `use_bias=true`. |
| `W_o` | [768, 768] | `attention.w_o_data` | Output projection. **Required.** |
| `b_o` | [768] | `attention.b_o_data` | Output projection bias. **Required.** |
| `alpha_q` | [num_heads] | `attention.alpha_q` | Optional per-head Q scaling. |
| `alpha_k` | [num_kv_heads] | `attention.alpha_k` | Optional per-head K scaling. |

### FFN Sub-layer (SwiGLU)

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_gate` | [768, 3072] | `ffn.w_gate_data` | Gate projection (SiLU branch of SwiGLU). |
| `W1` | [768, 3072] | `ffn.w1_data` | Up projection (linear branch). **Required.** |
| `W2` | [3072, 768] | `ffn.w2_data` | Down projection. **Required.** |
| `b2` | [1, 768] | `ffn.b2_data` | Down projection bias. **Required.** Stored as 2D `[1, d_model]`. |

### Normalization & Scaling

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `rms1_gamma` | [768] | `rms1.gamma` | Pre-attention RMSNorm scale. **Required.** |
| `rms2_gamma` | [768] | `rms2.gamma` | Pre-FFN RMSNorm scale. **Required.** |
| `rms_post_attn_gamma` | [768] | `rms_post_attn.gamma` | Post-attention RMSNorm (sandwich norm). **Deleted** — schema field retained for format compat, never populated. |
| `rms_post_ffn_gamma` | [768] | `rms_post_ffn.gamma` | Post-FFN RMSNorm (sandwich norm). **Deleted** — same as above. |
| `layer_scale1` | [1] | `layer_scale1` | LayerScale for attention residual. Conditional on `use_layer_scale`. |
| `layer_scale2` | [1] | `layer_scale2` | LayerScale for FFN residual. Conditional on `use_layer_scale`. |

### Per-Layer Parameter Count

~8.1M parameters per encoder layer (with biases), ~48.6M total for 6 layers.

---

## 3. LM Head

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `projection` | [vocab_size, 768] | `lm_head.projection_data` | **Aliased** from embedding `token_weights` when `tie_embeddings=true`. Only independently allocated when untied. |
| `bias` | [vocab_size] | `lm_head.bias_data` | Conditional on `use_bias`. |

---

## 4. Final RMSNorm

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `final_rms_gamma` | [768] | `final_rms_gamma` | Pre-LM-head RMSNorm. **Has weight decay (`wd_mult=1.0`) AND slow LR (`lr_mult=0.1`)**. Prevents unbounded logit temperature growth. |

---

## 5. ScratchBlock (Conditional: `scratch_block.enabled`)

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `atom_type_embeddings` | [3, 64] | `scratch_block.atom_type_embeddings` | Atom type lookup table. `NUM_ATOM_TYPES=3` (NONE, INT, FLOAT). `atom_embedding_dim=64`. |
| `atom_projection` | [64, 768] | `scratch_block.atom_projection` | Projects atom embeddings to hidden dim. |

---

## 6. ReasoningHead (Conditional: `reasoning_head.enabled`)

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_op` | [num_ops, 832] | `reasoning_head.w_op_data` | Operation logit projection. `d_total = d_model + atom_embedding_dim` = 768+64 = 832. `num_ops` hardcoded to 8 (not in config). |
| `b_op` | [1, num_ops] | `reasoning_head.b_op_data` | Operation bias. Stored as 2D `[1, num_ops]`. |
| `W_arg1` | [1, 832] | `reasoning_head.w_arg1_data` | Arg1 scoring vector. |
| `W_arg2` | [1, 832] | `reasoning_head.w_arg2_data` | Arg2 scoring vector. |

---

## 7. ExecutionBlock v2 (Conditional: `execution_block.enabled`)

Runtime config (`ai_config.json`): `V=8 slots, K=4 steps, num_ops=4, d_key=64, d_type=16`.

### Decode MLP

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `w_decode_1` | [24, 16] | `execution_block.w_decode_1_data` | Value decode MLP layer 1. |
| `b_decode_1` | [1, 16] | `execution_block.b_decode_1_data` | Decode MLP bias 1. Stored as 2D. |
| `w_decode_2` | [16, 1] | `execution_block.w_decode_2_data` | Value decode MLP layer 2. |

### Selection Projections

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `w_arg1_select` | [2304, 768] | `execution_block.w_arg1_select_data` | Arg1 selection. `3×d_model`. |
| `w_arg2_select` | [2304, 768] | `execution_block.w_arg2_select_data` | Arg2 selection. |
| `W_op_select` | [2304, 4] | `execution_block.w_op_select_data` | Op selection logits. |

### Memory Addressing

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_key_proj` | [768, 64] | `execution_block.w_key_proj_data` | Key generation from result embedding. |
| `W_write_query` | [3072, 64] | `execution_block.w_write_query_data` | Write-head query. `4×d_model → d_key`. |
| `W_write_key` | [64, 64] | `execution_block.w_write_key_data` | Write-head key. |
| `alpha` | [1, 1] | `execution_block.alpha_data` | Learned content score scalar. |
| `beta` | [1, 1] | `execution_block.beta_data` | Learned usage penalty scalar. |

### Embeddings & Type Tags

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `step_embeddings` | [4, 768] | `execution_block.step_embeddings_data` | Step encoding. `[K, d_model]`. |
| `type_num_embed` | [1, 16] | `execution_block.type_num_embed_data` | Type tag embedding. `[1, d_type]`. Stored as 2D. |

### Value Injection Path

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_value_to_emb` | [1, 768] | `execution_block.w_value_to_emb_data` | Linear scalar value → embedding. |
| `b_value_to_emb` | [1, 768] | `execution_block.b_value_to_emb_data` | Value → embedding bias. |
| `w_inject_gate` | [768, 1] | `execution_block.w_inject_gate_data` | Per-token injection gate. |

### Cross-Attention Read

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_Q_read` | [768, 64] | `execution_block.w_q_read_data` | Cross-attention query projection. |
| `W_K_read` | [64, 64] | `execution_block.w_k_read_data` | Cross-attention key projection. |
| `W_V_read` | [768, 64] | `execution_block.w_v_read_data` | Cross-attention value projection. |
| `W_O_read` | [64, 768] | `execution_block.w_o_read_data` | Cross-attention output projection. |
| `W_gate_read` | [768, 1] | `execution_block.w_gate_read_data` | Per-token read gate. |
| `tau` | [1, 1] | `execution_block.tau_data` | Learnable temperature for attention. |

### Trace Encoding

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `E_slot` | [8, 768] | `execution_block.e_slot_data` | Slot embedding for trace. `[V, d_model]`. |
| `E_op` | [4, 768] | `execution_block.e_op_data` | Op embedding for trace. `[num_ops, d_model]`. |
| `W_scal` | [3, 768] | `execution_block.w_scal_data` | Scalar projection (v1, v2, v_out). |
| `b_scal` | [1, 768] | `execution_block.b_scal_data` | Scalar projection bias. |
| `W_trace` | [3072, 768] | `execution_block.w_trace_data` | Flattened trace → d_model. `[K×768, 768]`. |
| `b_trace` | [1, 768] | `execution_block.b_trace_data` | Trace projection bias. |
| `W_reason_gate` | [1536, 768] | `execution_block.w_reason_gate_data` | Reasoning state update gate. `[2×768, 768]`. |
| `W_trace_gate` | [1536, 768] | `execution_block.w_trace_gate_data` | Trace gate logits. `[2×768, 768]`. |

---

## 8. DecodeTimeSlotSelector (Conditional: `slot_selector.enabled`)

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `W_q_select` | [768, 64] | `slot_selector.w_q_select_data` | Query projection for slot selection. `[d_model, d_selector]`. |
| `W_k_select` | [d_slot_features, 64] | `slot_selector.w_k_select_data` | Key projection from slot features. |
| `null_key_select` | [1, 64] | `slot_selector.null_key_select_data` | Learned NULL key (no-slot option). |
| `null_logit_bias` | [1] | `slot_selector.null_logit_bias_data` | Scalar NULL bias. |

---

## 9. Loss Weighting (Conditional)

| Parameter | Shape | FlatBuffers Field | Notes |
|-----------|-------|-------------------|-------|
| `log_var_text` | [1] | `loss_weighting.log_var_text` | Learned uncertainty for text loss (homoscedastic). |
| `log_var_numeric` | [1] | `loss_weighting.log_var_numeric` | Learned uncertainty for numeric loss. |

---

## 10. MTP Auxiliary Heads (Conditional: `mtp_enabled`, ×K heads)

Not serialized in the FlatBuffers schema. Per head:

| Parameter | Shape | Notes |
|-----------|-------|-------|
| `weight` | [vocab_size, 768] | Per auxiliary prediction head. |
| `bias` | [vocab_size] | Per auxiliary head bias. |

---

## Weight Tying & Aliasing Rules

| Condition | Effect |
|-----------|--------|
| `tie_embeddings=true` (default) | `EmbeddingLayer::token_weights_` and `LMHeadLayer::weights_` share the **same GPU buffer**. Gradients accumulate in-place (`g = g_lm + g_emb`). **Never zero both, add both to param groups, or free both.** |
| `positional_encoding != NONE` | `position_weights` is **not allocated**. ALiBi/RoPE inject position in attention, not embedding. |
| `use_bias=false` | All bias tensors (`b_qkv`, `b_o`, `b2`, LM head `bias`) are **not allocated**. |
| Sandwich norm **deleted** | `rms_post_attn_gamma` and `rms_post_ffn_gamma` are **never populated** — schema fields preserved for format forward-compat only. |

---

## Checkpoint Format

- **File identifier:** `GRMT`
- **File extension:** `.grmt`
- **Serialization:** FlatBuffers with post-write CRC32 + xxHash64 verification
- **Schema:** `resources/models/GRIM-text/training/schemas/grim_transformer_model.fbs`
- **Numeric head:** Schema field exists (`numeric_head`) but is **never populated** (layer deleted Issue #142).
