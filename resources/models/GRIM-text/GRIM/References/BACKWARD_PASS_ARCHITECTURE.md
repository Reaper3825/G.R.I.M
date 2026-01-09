# GRIM-text Backward Pass Architecture

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Author:** GRIM Development Team  
**Audience:** Technical reviewers, professor evaluation

---

## Executive Summary

The GRIM-text backward pass implements gradient computation through a **3-phase reverse pipeline** that mirrors the forward pass in opposite order. The architecture computes gradients for 104M+ parameters across embedding, encoder, and output layers using GPU-accelerated operations with Flash Attention v2 backward kernels, GQA-aware gradient accumulation, and BF16/FP32 mixed precision.

### Key Architectural Decisions

1. **3-Phase Reverse Structure**: Output (Phase 1) → Encoder (Phase 2) → Input (Phase 3)
2. **Unified Loss Integration**: Cross-entropy gradient computed during forward pass, NOT recomputed
3. **GQA Gradient Dimensions**: Q gradients use `num_heads=12`, K/V use `num_kv_heads=4`
4. **Flash Attention v2 BF16**: All attention gradients computed in BF16 with FP32 conversion
5. **RoPE Inverse Rotation**: Gradients transformed back to original coordinate space
6. **Fail-Loud Policy**: No fallbacks—crash immediately with detailed error on invalid state

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKWARD PASS ORCHESTRATION                          │
│                                                                             │
│    ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐        │
│    │   PHASE 1    │───▶│     PHASE 2     │───▶│      PHASE 3      │        │
│    │ Output Layer │    │ Encoder Layers  │    │   Input Layer     │        │
│    │   (LM Head)  │    │   (L11 → L0)    │    │   (Embedding)     │        │
│    └──────────────┘    └─────────────────┘    └───────────────────┘        │
│           │                    │                      │                     │
│    ┌──────▼──────┐    ┌───────▼───────┐    ┌────────▼────────┐             │
│    │ grad_logits │    │ Per-Layer     │    │ ScratchBlock    │             │
│    │ LM Head bwd │    │ 8-Step Loop   │    │ Embedding bwd   │             │
│    │ NumericHead │    │ FFN→Attn→Res  │    │ tie_embeddings  │             │
│    └─────────────┘    └───────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Flow Summary

| Phase | Name | Input | Output | Key Operations |
|-------|------|-------|--------|----------------|
| **1** | Output Layer | `grad_logits` (from UnifiedLoss) | `grad_encoder_out` | LM Head backward, NumericHead backward |
| **2** | Encoder | `grad_encoder_out` | `grad_embedding_out` | 12 layers × 8-step backward, Flash Attn v2 bwd |
| **3** | Input Layer | `grad_embedding_out` | `grad_embeddings` | ScratchBlock backward, Embedding scatter-add |

---

## 1. BackwardContext Architecture

### 1.1 Central State Structure

All backward operations receive a shared `BackwardContext` struct containing configuration, training state, and gradient flow state:

```cpp
struct BackwardContext {
    //=== Configuration ===//
    const ModelConfig* config;
    TrainingState_GPU* training_state;
    
    //=== Batch Dimensions ===//
    int batch_size;
    int seq_len;
    int total_tokens;  // batch_size × seq_len
    
    //=== Gradient Flow State ===//
    float* current_grad;       // Propagating gradient buffer
    float grad_scale;          // Per-token normalization factor
    bool accumulate;           // Add to existing gradients (beta=1) vs overwrite (beta=0)
    
    //=== cuBLAS Constants ===//
    cublasHandle_t cublas_handle;
    float alpha;               // Always 1.0
    float beta_zero;           // 0.0 for overwrite
    float beta_accum;          // 1.0 for accumulation (or 0.0 if !accumulate)
    
    //=== External Components ===//
    GrimTokenizer* tokenizer;
    
    //=== Diagnostics ===//
    int backward_call_id;      // Unique ID for logging
    bool enable_grad_checks;   // Validate gradient stats
    float explosion_threshold; // Max gradient norm before FATAL
    
    //=== Phase Status ===//
    BackwardStatus phase1_status;
    BackwardStatus phase2_status;
    BackwardStatus phase3_status;
};
```

### 1.2 BackwardStatus Error Codes

The backward pass uses explicit status codes for fail-loud error handling:

```cpp
enum class BackwardStatus : int {
    SUCCESS = 0,
    FATAL_ERROR = 1,
    GRADIENT_EXPLOSION = 2,
    INVALID_STATE = 3,
    CUDA_ERROR = 4,
    CUBLAS_ERROR = 5,
    TENSOR_CONTRACT_VIOLATION = 6,
    NULL_POINTER = 7
};
```

### 1.3 Fail-Loud Macros

```cpp
#define BWD_CHECK_PTR(ctx, ptr, name, layer)                                    \
    if (!(ptr)) {                                                               \
        LOG_FATAL("[Backward] " << name << " is NULL at layer " << layer);      \
        return BackwardStatus::NULL_POINTER;                                    \
    }

#define BWD_CHECK_CUDA(ctx, call, operation, layer)                             \
    { cudaError_t err = (call);                                                 \
      if (err != cudaSuccess) {                                                 \
          LOG_FATAL("[Backward] CUDA error in " << operation << " L" << layer   \
                    << ": " << cudaGetErrorString(err));                        \
          return BackwardStatus::CUDA_ERROR;                                    \
      }                                                                         \
    }

#define BWD_CHECK_CUBLAS(ctx, call, operation, layer)                           \
    { cublasStatus_t status = (call);                                           \
      if (status != CUBLAS_STATUS_SUCCESS) {                                    \
          LOG_FATAL("[Backward] cuBLAS error in " << operation << " L" << layer \
                    << ": status=" << status);                                  \
          return BackwardStatus::CUBLAS_ERROR;                                  \
      }                                                                         \
    }

#define BWD_FAIL_LOUD(ctx, status, message, layer)                              \
    { LOG_FATAL("[Backward] " << message << " at layer " << layer);             \
      return (status);                                                          \
    }
```

---

## 2. Phase 1: Output Layer Backward

### 2.1 Overview

Phase 1 computes gradients for the output layer, receiving pre-computed `grad_logits` from the UnifiedLoss system (not recomputed).

**Location:** `BackwardPhase1_OutputLayer.cu`

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: OUTPUT LAYER                    │
│                                                             │
│   UnifiedLoss (forward)                                     │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────┐                                          │
│   │ grad_logits │  (Skip CE recomputation!)                │
│   └──────┬──────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────────────────────────────┐                  │
│   │         LM HEAD BACKWARD            │                  │
│   │  grad_W_lm = grad_logits^T @ enc_out│                  │
│   │  grad_enc = grad_logits @ W_lm^T    │                  │
│   └──────────────┬──────────────────────┘                  │
│                  │                                          │
│                  ▼                                          │
│          ctx.current_grad = grad_encoder_out               │
│                  │                                          │
│                  ▼                                          │
│           [To Phase 2]                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Step-by-Step Execution

#### Step 1: Cross-Entropy Gradient (SKIP)

**Critical Design Decision:** The cross-entropy gradient is computed during the forward pass by UnifiedLoss, NOT recomputed here. This avoids:
- Redundant softmax computation
- Numerical precision differences between forward/backward
- Double the GPU memory for logits

```cpp
// grad_logits already contains: (softmax(logits) - one_hot_target) / valid_tokens
// Written by UnifiedLoss during forward pass
float* grad_logits = ts->loss_params.grad_logits;
BWD_CHECK_PTR(ctx, grad_logits, "grad_logits (from UnifiedLoss)", 0);
```

#### Step 2: Apply Gradient Scale

```cpp
if (ctx.grad_scale != 1.0f) {
    launchScaleGradients(grad_logits, ctx.grad_scale, 
                         total_tokens * cfg->vocab_size, stream);
}
```

#### Step 3: LM Head Backward

Computes weight gradients and propagates to encoder output:

```cpp
// LM Head Weight Gradient: grad_W = grad_logits^T @ encoder_output
// Shape: [vocab_size, d_model] = [total_tokens, vocab_size]^T @ [total_tokens, d_model]
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_model, cfg->vocab_size, total_tokens,
    &alpha,
    encoder_output, cfg->d_model,
    grad_logits, cfg->vocab_size,
    &beta_accum,
    grad_W_lm_head, cfg->d_model);

// Propagate to Encoder Output: grad_enc = grad_logits @ W_lm^T
// Shape: [total_tokens, d_model] = [total_tokens, vocab_size] @ [vocab_size, d_model]^T
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    cfg->d_model, total_tokens, cfg->vocab_size,
    &alpha,
    W_lm_head, cfg->d_model,         // Transposed: [d_model, vocab_size]^T
    grad_logits, cfg->vocab_size,
    &beta_zero,
    grad_encoder_out, cfg->d_model);

ctx.current_grad = grad_encoder_out;
```

#### Step 4: NumericHead Backward (Optional)

If numeric prediction is enabled:

```cpp
if (cfg->use_numeric_head) {
    launchNumericHeadBackward(
        grad_numeric_preds,     // From numeric loss
        encoder_output,
        grad_W_numeric,
        grad_b_numeric,
        ctx.current_grad,       // Accumulate into encoder gradient
        total_tokens,
        cfg->d_model,
        cfg->numeric_output_dim,
        stream);
}
```

---

## 3. Phase 2: Encoder Backward

### 3.1 Overview

Phase 2 is the most complex phase, iterating through encoder layers **in reverse order** (L11 → L0). Each layer executes an 8-step backward process.

**Location:** `BackwardPhase2_Encoder.cu` (1357 lines)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2: ENCODER BACKWARD                            │
│                                                                             │
│        ┌─────────────────────────────────────────────────────────┐         │
│        │              LAYER LOOP (L11 → L0)                      │         │
│        │                                                         │         │
│        │   for (int layer = num_layers - 1; layer >= 0; --layer) │         │
│        │   {                                                     │         │
│        │       executeLayerBackward(ctx, layer);                 │         │
│        │   }                                                     │         │
│        └─────────────────────────────────────────────────────────┘         │
│                                                                             │
│   PER-LAYER 8-STEP PROCESS:                                                │
│                                                                             │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                     │
│   │ Step 1 │───▶│ Step 2 │───▶│ Step 3 │───▶│ Step 4 │                     │
│   │  Copy  │    │Reconst.│    │  FFN   │    │ RMS2   │                     │
│   │  Grad  │    │Residual│    │Backward│    │Backward│                     │
│   └────────┘    └────────┘    └────────┘    └────────┘                     │
│        │                                                                    │
│        ▼                                                                    │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐                     │
│   │ Step 5 │───▶│ Step 6 │───▶│ Step 7 │───▶│ Step 8 │                     │
│   │ Merge  │    │  Attn  │    │ RMS1   │    │ Final  │                     │
│   │FFN+Res │    │Backward│    │Backward│    │ Merge  │                     │
│   └────────┘    └────────┘    └────────┘    └────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Pre-Norm Transformer Backward Math

The GRIM-text encoder uses **pre-norm** architecture (LayerNorm before sublayer):

**Forward (per layer):**
```
residual1 = Attention(RMSNorm1(layer_input)) + layer_input    [Skip Connection 1]
layer_output = FFN(RMSNorm2(residual1)) + residual1          [Skip Connection 2]
```

**Backward (chain rule):**
```
grad_residual1 = grad_layer_output + grad_through_FFN         [From Skip 2]
grad_layer_input = grad_residual1 + grad_through_Attention    [From Skip 1]

Expanding:
grad_layer_input = grad_layer_output + grad_through_FFN + grad_through_Attention
```

### 3.3 Step-by-Step Per-Layer Backward

#### Step 1: Copy Current Gradient for FFN Path

```cpp
// Save grad_layer_output for FFN backward path
float* grad_ffn_input = ts->grad_ffn_input;
cudaMemcpyAsync(grad_ffn_input, ctx.current_grad,
                total_tokens * d_model * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
```

#### Step 2: Reconstruct Residual1

During forward, we stored `attn_output` but need `residual1 = attn_output + layer_input`:

```cpp
float* residual1 = ts->residual1_buffer;
launchResidualAdd(cached_attn_output, cached_layer_input, residual1,
                  total_tokens * d_model, stream);
```

#### Step 3: FFN Backward (W2 → GELU → W1)

```
FFN Forward:
    hidden = layer_input @ W1 + b1           [Projection up: d_model → d_ff]
    post_gelu = GELU(hidden)                 [Activation]
    output = post_gelu @ W2 + b2             [Projection down: d_ff → d_model]

FFN Backward:
    grad_W2 = post_gelu^T @ grad_output      [Weight gradient]
    grad_hidden = grad_output @ W2^T         [Input gradient through W2]
    grad_pre_gelu = grad_hidden ⊙ GELU'(hidden)  [GELU backward]
    grad_W1 = layer_input^T @ grad_pre_gelu  [Weight gradient]
    grad_input = grad_pre_gelu @ W1^T        [Input gradient through W1]
```

**Implementation:**

```cpp
// Step 3a: W2 weight gradient
// grad_W2 = cached_ffn_hidden^T @ grad_ffn_output
// Shape: [d_ff, d_model] = [total_tokens, d_ff]^T @ [total_tokens, d_model]
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_ff, cfg->d_model, total_tokens,
    &alpha,
    cached_ffn_hidden, cfg->d_ff,        // Post-GELU activation
    grad_ffn_output, cfg->d_model,
    &beta_accum,
    ts->ffn_w2_grads[layer], cfg->d_ff);

// Step 3b: Gradient through W2
// grad_hidden = grad_ffn_output @ W2
// Shape: [total_tokens, d_ff] = [total_tokens, d_model] @ [d_model, d_ff]
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    cfg->d_ff, total_tokens, cfg->d_model,
    &alpha,
    W2, cfg->d_ff,
    grad_ffn_output, cfg->d_model,
    &beta_zero,
    grad_ffn_hidden, cfg->d_ff);

// Step 3c: GELU Backward
// grad_pre_gelu = grad_hidden ⊙ GELU'(pre_gelu_hidden)
launchGELUBackward(grad_ffn_hidden, cached_pre_gelu_hidden,
                   grad_pre_gelu, total_tokens * cfg->d_ff, stream);

// Step 3d: W1 weight gradient
// grad_W1 = ln2_output^T @ grad_pre_gelu
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_model, cfg->d_ff, total_tokens,
    &alpha,
    cached_ln2_output, cfg->d_model,
    grad_pre_gelu, cfg->d_ff,
    &beta_accum,
    ts->ffn_w1_grads[layer], cfg->d_model);

// Step 3e: Gradient through W1
// grad_ffn_input = grad_pre_gelu @ W1
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    cfg->d_model, total_tokens, cfg->d_ff,
    &alpha,
    W1, cfg->d_model,
    grad_pre_gelu, cfg->d_ff,
    &beta_zero,
    grad_ffn_input, cfg->d_model);
```

#### Step 4: RMSNorm2 Backward

```cpp
launchRMSNormBackward(
    residual1,              // Input to RMSNorm2
    grad_ffn_input,         // grad_output (modified in-place)
    gamma2,                 // Scale parameter
    grad_ffn_input,         // grad_input (overwrite)
    ts->rms2_gamma_grads[layer],  // grad_gamma
    total_tokens,
    cfg->d_model,
    cfg->rms_norm_eps,
    stream);
```

**RMSNorm Backward Math:**
```
Forward: y = x * inv_rms * gamma
         inv_rms = 1 / sqrt(mean(x²) + eps)

Backward:
    grad_gamma = sum(grad_output ⊙ x * inv_rms)
    
    grad_input = grad_output ⊙ gamma * inv_rms 
                 - x * inv_rms * mean(grad_output ⊙ gamma ⊙ x * inv_rms)
```

#### Step 5: Merge FFN Path + Residual

```cpp
// grad_attn_input = grad_ffn_input + ctx.current_grad
// This is grad_residual1 from the chain rule expansion
launchResidualAdd(grad_ffn_input, ctx.current_grad, grad_attn_input,
                  total_tokens * cfg->d_model, stream);
ctx.current_grad = grad_attn_input;
```

#### Step 6: Attention Backward (Most Complex)

The attention backward has 7 sub-steps:

##### Step 6.1: W_o Weight Gradient

```cpp
// Flatten attention output from BHSD to flat format for W_o gradient
// grad_W_o = cached_attn_bhsd^T @ grad_attn_output
TensorContract::convert(cached_attn_bhsd, grad_attn_out_flat, ...);

cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_model, cfg->d_model, total_tokens,
    &alpha,
    grad_attn_out_flat, cfg->d_model,    // Flattened attention output
    ctx.current_grad, cfg->d_model,
    &beta_accum,
    ts->attn_wo_weight_grads[layer], cfg->d_model);
```

##### Step 6.2: Gradient Through W_o

```cpp
// grad_attn_flat = grad_attn_output @ W_o
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    cfg->d_model, total_tokens, cfg->d_model,
    &alpha,
    W_o, cfg->d_model,
    ctx.current_grad, cfg->d_model,
    &beta_zero,
    grad_attn_out_flat, cfg->d_model);
```

##### Step 6.3: Reshape to BHSD for Flash Attention

```cpp
// Convert flat gradient to [batch, num_heads, seq, head_dim] format
TensorContract::convert(grad_attn_out_flat, grad_attn_out_reshaped,
    {total_tokens, d_model},
    {batch_size, num_heads, seq_len, head_dim});
```

##### Step 6.4: Flash Attention v2 Backward

This is the core attention backward using NVIDIA's Flash Attention:

```cpp
// Convert FP32 → BF16 for Flash Attention kernels
TensorConversion::convert_BHSD_to_BSHD_bf16(cached_Q, fa_q, ...);
TensorConversion::convert_BHSD_to_BSHD_bf16(cached_K, fa_k, ...);
TensorConversion::convert_BHSD_to_BSHD_bf16(cached_V, fa_v, ...);
TensorConversion::convert_BHSD_to_BSHD_bf16(cached_attn_bhsd, fa_out, ...);
TensorConversion::convert_BHSD_to_BSHD_bf16(grad_attn_out_reshaped, fa_dout, ...);

// Execute Flash Attention backward
flash_attn_bwd_ex(
    fa_q, fa_k, fa_v,           // Cached forward inputs (BF16)
    fa_out, fa_dout,            // Output and gradient (BF16)
    cached_softmax_lse,         // Log-sum-exp from forward
    alibi_slopes,               // ALiBi positional bias slopes
    fa_dq, fa_dk, fa_dv,        // Output gradients (BF16)
    fa_dq_accum,                // Workspace for dQ accumulation
    fa_dsoftmax_sum,            // Workspace for softmax gradient
    batch_size, seq_len,
    num_heads, num_kv_heads,    // GQA dimensions
    head_dim,
    causal_mask,                // true for autoregressive
    deterministic,              // true for reproducibility
    stream);

// Convert BF16 → FP32 for subsequent operations
TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dq, grad_Q, ...);  // [B, H, S, D]
TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dk, grad_K, ...);  // [B, KV, S, D]
TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dv, grad_V, ...);  // [B, KV, S, D]
```

**GQA Gradient Dimensions:**
```
grad_Q: [batch_size, num_heads=12, seq_len, head_dim=64]
grad_K: [batch_size, num_kv_heads=4, seq_len, head_dim=64]
grad_V: [batch_size, num_kv_heads=4, seq_len, head_dim=64]
```

##### Step 6.5: RoPE Inverse Rotation

**Critical:** Flash Attention computed gradients in ROTATED space. We must transform back to original space:

```cpp
// Apply inverse rotation (-θ) to Q and K gradients
PBM::launchRoPERotationGQA_backward(
    grad_Q,                          // [B, H, S, D] - in-place
    grad_K,                          // [B, KV, S, D] - in-place
    rope_inv_freq,                   // Inverse frequencies [rotary_dim/2]
    batch_size,
    num_heads,                       // Q head count = 12
    num_kv_heads,                    // K head count = 4
    seq_len,
    head_dim,
    rotary_dim,
    stream);
```

**RoPE Backward Math:**
```
Forward: Q_rot = Q ⊙ cos(θ) + rotate_half(Q) ⊙ sin(θ)
Backward: grad_Q = grad_Q_rot ⊙ cos(θ) - rotate_half(grad_Q_rot) ⊙ sin(θ)
                 = grad_Q_rot ⊙ cos(-θ) + rotate_half(grad_Q_rot) ⊙ sin(-θ)
```

##### Step 6.6: Merge Q, K, V Gradients

```cpp
// Merge GQA gradients into fused [total_tokens, total_qkv_dim] format
// total_qkv_dim = d_model + 2*kv_dim = 768 + 2*256 = 1280
TensorContract::merge_qkv_grads_gqa(
    view_grad_Q,    // [B, 12, S, 64]
    view_grad_K,    // [B, 4, S, 64]
    view_grad_V,    // [B, 4, S, 64]
    view_grad_qkv,  // [total_tokens, 1280]
    gqa_dims,
    stream);
```

##### Step 6.7: QKV Weight Gradient + Propagation

```cpp
// QKV weight gradient: grad_W_qkv = ln1_output^T @ grad_qkv
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_model, total_qkv_dim, total_tokens,
    &alpha,
    cached_ln1_output, cfg->d_model,
    grad_qkv_concat, total_qkv_dim,
    &beta_accum,
    ts->attn_qkv_weight_grads[layer], cfg->d_model);

// Propagate through QKV projection: grad_ln1 = grad_qkv @ W_qkv^T
cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    cfg->d_model, total_tokens, total_qkv_dim,
    &alpha,
    W_qkv, total_qkv_dim,
    grad_qkv_concat, total_qkv_dim,
    &beta_zero,
    grad_qkv_input, cfg->d_model);
```

#### Step 7: RMSNorm1 Backward

```cpp
launchRMSNormBackward(
    cached_layer_input,         // Input to RMSNorm1
    grad_qkv_input,             // grad_output
    gamma1,                     // Scale parameter
    grad_rms1_output,           // grad_input
    ts->rms1_gamma_grads[layer],  // grad_gamma
    total_tokens,
    cfg->d_model,
    cfg->rms_norm_eps,
    stream);
```

#### Step 8: Final Residual Add (Skip Connection 1)

```cpp
// grad_layer_input = grad_rms1_output + ctx.current_grad
// This completes: grad_input = grad_through_attn + grad_residual1
launchResidualAdd(grad_rms1_output, ctx.current_grad, ctx.current_grad,
                  total_tokens * cfg->d_model, stream);
```

---

## 4. Phase 3: Input Layer Backward

### 4.1 Overview

Phase 3 computes gradients for the embedding layer and handles ScratchBlock backward.

**Location:** `BackwardPhase3_InputLayer.cu` (267 lines)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: INPUT LAYER                                 │
│                                                                             │
│   ctx.current_grad                                                          │
│         │ (from Phase 2)                                                    │
│         ▼                                                                   │
│   ┌─────────────────────────────────────────┐                              │
│   │        SCRATCHBLOCK BACKWARD            │  (if enabled)                │
│   │  Process atom tokens → raw embedding    │                              │
│   └──────────────┬──────────────────────────┘                              │
│                  │                                                          │
│                  ▼                                                          │
│   ┌─────────────────────────────────────────┐                              │
│   │         EMBEDDING BACKWARD              │                              │
│   │   scatter-add: atomicAdd for each token │                              │
│   │   grad_E[token_id] += grad_output[pos]  │                              │
│   └──────────────┬──────────────────────────┘                              │
│                  │                                                          │
│                  ▼                                                          │
│   ┌─────────────────────────────────────────┐                              │
│   │       TIE_EMBEDDINGS MERGE              │                              │
│   │  (Skip if buffers aliased)              │                              │
│   │  grad_E_final = grad_E + grad_lm_head   │                              │
│   └─────────────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Step 1: ScratchBlock Backward (Optional)

If ScratchBlock reasoning is enabled, process atom token gradients:

```cpp
if (cfg->scratch_block.enabled) {
    // ScratchBlock backward processes cached atom information
    // and propagates gradients through atom detection to raw embeddings
    ScratchBlock::backward(
        ctx.current_grad,           // Gradient from encoder
        cached_atom_info,           // From forward pass
        grad_scratch_output,        // Output
        total_tokens,
        cfg->d_model,
        stream);
}
```

### 4.3 Step 2: Embedding Backward (Scatter-Add)

The embedding backward is unique because it's a **sparse update**—each token contributes gradients only to its row in the embedding matrix:

```cpp
// For each position p with token_id t:
// grad_embedding[t] += grad_output[p]

__global__ void embeddingBackwardKernel(
    const float* grad_output,      // [total_tokens, d_model]
    const int* token_ids,          // [total_tokens]
    float* grad_embedding,         // [vocab_size, d_model]
    int total_tokens,
    int d_model
) {
    int pos = blockIdx.x;          // Position in sequence
    int dim = threadIdx.x;         // Dimension index
    
    if (pos < total_tokens && dim < d_model) {
        int token_id = token_ids[pos];
        // Atomic add because multiple positions may share same token_id
        atomicAdd(&grad_embedding[token_id * d_model + dim],
                  grad_output[pos * d_model + dim]);
    }
}
```

**Launch:**
```cpp
launchEmbeddingBackward(
    ctx.current_grad,           // [total_tokens, d_model]
    ts->input_token_ids,        // [total_tokens]
    ts->embedding_grads,        // [vocab_size, d_model]
    total_tokens,
    cfg->d_model,
    stream);
```

### 4.4 Step 3: Tie Embeddings Handling

When `tie_embeddings=true`, the embedding and LM head share the same weight matrix. The gradients from both paths must be combined:

```cpp
if (cfg->tie_embeddings) {
    // Check if buffers are already aliased (same pointer)
    if (ts->embedding_grads == ts->lm_head_weight_grads) {
        // ALIASED: atomicAdd already accumulated both gradients
        // Skip merge—gradients are already combined
        LOG_DEBUG("tie_embeddings: buffers aliased, skipping merge");
    } else {
        // SEPARATE: must merge explicitly
        // This happens if someone allocated separate buffers incorrectly
        launchResidualAdd(
            ts->embedding_grads,
            ts->lm_head_weight_grads,
            ts->embedding_grads,  // In-place merge
            vocab_size * d_model,
            stream);
        LOG_DEBUG("tie_embeddings: merged separate buffers");
    }
}
```

---

## 5. Flash Attention v2 Backward Details

### 5.1 Parameter Structure

```cpp
struct Flash_bwd_params : public Flash_fwd_params {
    // === Gradient Output Pointers ===
    void* __restrict__ do_ptr;         // dOut: [B, S, H, D] BF16
    void* __restrict__ dq_ptr;         // dQ: [B, S, H, D] BF16
    void* __restrict__ dk_ptr;         // dK: [B, S, KV, D] BF16
    void* __restrict__ dv_ptr;         // dV: [B, S, KV, D] BF16
    
    // === Workspace Buffers ===
    void* __restrict__ dq_accum_ptr;   // [B, H, S, D] FP32 accumulator
    void* __restrict__ dsoftmax_sum;   // [B, H, S] FP32 softmax grad sum
    
    // === Flags ===
    bool deterministic;                // Use deterministic algorithm
};
```

### 5.2 Workspace Requirements

```cpp
// dq_accum: For accumulating dQ across KV blocks
size_t dq_accum_bytes() {
    return batch_size * num_heads * seq_len * head_dim * sizeof(float);
}

// dsoftmax_sum: For softmax gradient computation
size_t dsoftmax_sum_bytes() {
    return batch_size * num_heads * seq_len * sizeof(float);
}
```

### 5.3 BF16 Conversion

Flash Attention v2 requires BF16 input format with specific layout:

| Buffer | Forward Layout | Backward Layout |
|--------|---------------|-----------------|
| Q, K, V | BHSD → BSHD (BF16) | Same |
| Output | BHSD → BSHD (BF16) | Same |
| Softmax LSE | [B, H, S] (FP32) | Reused from forward |
| dQ, dK, dV | BSHD (BF16) → BHSD (FP32) | Converted back |

---

## 6. Gradient Statistics and Explosion Detection

### 6.1 GradStats Queue System

To avoid GPU-CPU synchronization during backward, gradients are checked via a deferred queue:

```cpp
struct GradCheckResult {
    float rms;          // Root mean square
    float max_abs;      // Maximum absolute value
    bool has_nan;       // Contains NaN
    bool has_inf;       // Contains Inf
    bool is_explosion;  // Exceeds threshold
};

// Queue gradient check (async, no sync)
void queueGradStats(
    const char* name,
    int layer,
    const float* grad,
    size_t count,
    float threshold,
    cudaStream_t stream);

// Flush queue at end of backward (single sync point)
void flushGradStats();
```

### 6.2 Explosion Detection

```cpp
// Check during gradient validation
if (ctx.enable_grad_checks) {
    auto result = computeGradStats(grad_buffer, count, stream);
    
    if (result.has_nan || result.has_inf) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::GRADIENT_EXPLOSION,
                      "NaN/Inf detected in gradients", layer);
    }
    
    if (result.max_abs > ctx.explosion_threshold) {
        LOG_WARNING("[Backward] Gradient explosion: max=" << result.max_abs
                    << " > threshold=" << ctx.explosion_threshold);
        // May continue or abort depending on config
    }
}
```

---

## 7. Memory Layout Reference

### 7.1 Gradient Buffer Sizes

| Buffer | Shape | Size (bytes) | Notes |
|--------|-------|--------------|-------|
| `grad_logits` | [tokens, vocab] | tokens × 37555 × 4 | From UnifiedLoss |
| `grad_encoder_out` | [tokens, d_model] | tokens × 768 × 4 | Phase 1 output |
| `grad_qkv_concat` | [tokens, total_qkv] | tokens × 1280 × 4 | GQA format |
| `grad_Q` | [B, H, S, D] | B × 12 × S × 64 × 4 | num_heads=12 |
| `grad_K` | [B, KV, S, D] | B × 4 × S × 64 × 4 | num_kv_heads=4 |
| `grad_V` | [B, KV, S, D] | B × 4 × S × 64 × 4 | num_kv_heads=4 |
| `embedding_grads` | [vocab, d_model] | 37555 × 768 × 4 | Sparse updates |

### 7.2 Weight Gradient Buffers (Per Layer)

| Buffer | Shape | Size | cuBLAS Beta |
|--------|-------|------|-------------|
| `attn_qkv_weight_grads` | [total_qkv, d_model] | 1280 × 768 × 4 | Accumulate |
| `attn_wo_weight_grads` | [d_model, d_model] | 768 × 768 × 4 | Accumulate |
| `ffn_w1_grads` | [d_ff, d_model] | 3072 × 768 × 4 | Accumulate |
| `ffn_w2_grads` | [d_model, d_ff] | 768 × 3072 × 4 | Accumulate |
| `rms1_gamma_grads` | [d_model] | 768 × 4 | atomicAdd |
| `rms2_gamma_grads` | [d_model] | 768 × 4 | atomicAdd |

---

## 8. cuBLAS Operation Summary

### 8.1 Matrix Multiplication Patterns

| Operation | Formula | cuBLAS Call |
|-----------|---------|-------------|
| Weight Grad | `dW = A^T @ dY` | `GEMM(OP_N, OP_T, ...)` |
| Input Grad | `dX = dY @ W^T` | `GEMM(OP_T, OP_N, ...)` |
| Fused Projection | `dX = dY @ W` | `GEMM(OP_N, OP_N, ...)` |

### 8.2 Beta Values

```cpp
float alpha = 1.0f;
float beta_zero = 0.0f;           // Overwrite output
float beta_accum = accumulate ? 1.0f : 0.0f;  // Add to existing

// Weight gradients: Always accumulate (may be called multiple times)
cublasSgemm(..., &alpha, ..., &beta_accum, weight_grads, ...);

// Input gradients: Overwrite (single computation per buffer)
cublasSgemm(..., &alpha, ..., &beta_zero, input_grads, ...);
```

---

## 9. Error Handling and Recovery

### 9.1 Phase Status Tracking

```cpp
BackwardStatus executeBackward(BackwardContext& ctx) {
    // Phase 1: Output Layer
    ctx.phase1_status = executePhase1_OutputLayer(ctx);
    if (ctx.phase1_status != BackwardStatus::SUCCESS) {
        LOG_FATAL("Phase 1 failed: " << getStatusString(ctx.phase1_status));
        return ctx.phase1_status;
    }
    
    // Phase 2: Encoder Layers
    ctx.phase2_status = executePhase2_Encoder(ctx);
    if (ctx.phase2_status != BackwardStatus::SUCCESS) {
        LOG_FATAL("Phase 2 failed: " << getStatusString(ctx.phase2_status));
        return ctx.phase2_status;
    }
    
    // Phase 3: Input Layer
    ctx.phase3_status = executePhase3_InputLayer(ctx);
    if (ctx.phase3_status != BackwardStatus::SUCCESS) {
        LOG_FATAL("Phase 3 failed: " << getStatusString(ctx.phase3_status));
        return ctx.phase3_status;
    }
    
    return BackwardStatus::SUCCESS;
}
```

### 9.2 Error Report Generation

```cpp
std::string getBackwardErrorReport(const BackwardContext& ctx) {
    std::ostringstream ss;
    ss << "=== BACKWARD PASS ERROR REPORT ===\n";
    ss << "Call ID: " << ctx.backward_call_id << "\n";
    ss << "Phase 1 (Output): " << getStatusString(ctx.phase1_status) << "\n";
    ss << "Phase 2 (Encoder): " << getStatusString(ctx.phase2_status) << "\n";
    ss << "Phase 3 (Input): " << getStatusString(ctx.phase3_status) << "\n";
    ss << "Batch size: " << ctx.batch_size << "\n";
    ss << "Seq len: " << ctx.seq_len << "\n";
    ss << "Grad scale: " << ctx.grad_scale << "\n";
    return ss.str();
}
```

---

## 10. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `BackwardOps_Orchestrator.hpp` | 148 | Entry point declarations |
| `BackwardOps_Orchestrator.cu` | 349 | Phase orchestration, timing |
| `BackwardContext.hpp` | 293 | Context struct, status enum, macros |
| `BackwardPhase1_OutputLayer.cu` | 311 | LM Head backward, NumericHead |
| `BackwardPhase2_Encoder.cu` | 1357 | 12-layer × 8-step backward |
| `BackwardPhase3_InputLayer.cu` | 267 | Embedding backward, tie_embeddings |
| `Flash_Attention_Kernal.cu` | 981 | Flash Attention v2 backward |
| `RMSNorm_Kernel_GPU.cu` | 248 | RMSNorm backward kernel |

---

## Appendix A: Complete Backward Pass Sequence

```
executeBackward(ctx)
│
├── PHASE 1: Output Layer (311 lines)
│   ├── Skip CE recomputation (grad_logits from UnifiedLoss)
│   ├── Apply grad_scale
│   ├── LM Head backward
│   │   ├── grad_W_lm = grad_logits^T @ encoder_out
│   │   └── grad_enc = grad_logits @ W_lm^T
│   ├── NumericHead backward (optional)
│   └── ctx.current_grad = grad_encoder_out
│
├── PHASE 2: Encoder Layers (1357 lines)
│   └── for layer in [11, 10, ..., 0]:
│       ├── Step 1: Copy grad for FFN path
│       ├── Step 2: Reconstruct residual1
│       ├── Step 3: FFN Backward
│       │   ├── grad_W2 = ffn_hidden^T @ grad_ffn
│       │   ├── grad_hidden = grad_ffn @ W2^T
│       │   ├── GELU backward
│       │   ├── grad_W1 = ln2_out^T @ grad_pre_gelu
│       │   └── grad_ffn_in = grad_pre_gelu @ W1^T
│       ├── Step 4: RMSNorm2 backward
│       ├── Step 5: Merge FFN + residual paths
│       ├── Step 6: Attention Backward
│       │   ├── W_o weight gradient
│       │   ├── Gradient through W_o
│       │   ├── Reshape to BHSD
│       │   ├── FP32→BF16 conversion
│       │   ├── Flash Attention v2 backward
│       │   ├── BF16→FP32 conversion
│       │   ├── RoPE inverse rotation
│       │   ├── Merge Q/K/V gradients
│       │   ├── QKV weight gradient
│       │   └── Propagate through QKV projection
│       ├── Step 7: RMSNorm1 backward
│       └── Step 8: Final residual add
│
└── PHASE 3: Input Layer (267 lines)
    ├── ScratchBlock backward (optional)
    ├── Embedding backward (atomic scatter-add)
    └── tie_embeddings merge (if not aliased)
```

---

## Appendix B: Performance Considerations

### B.1 Memory Traffic

| Operation | Read | Write | Notes |
|-----------|------|-------|-------|
| LM Head backward | grad_logits + enc_out | grad_W + grad_enc | Large vocab |
| Flash Attn backward | Q,K,V,O,dO,LSE | dQ,dK,dV | BF16, tiled |
| Embedding backward | grad_out + token_ids | grad_embed | Sparse, atomic |

### B.2 Synchronization Points

1. **Phase boundaries** - Implicit sync via gradient buffer dependencies
2. **Flash Attention** - Workspace buffers must be ready
3. **Gradient checks** - Optional sync for validation
4. **Final flush** - Single sync at end for deferred stats

### B.3 cuBLAS Stream Overlap

All cuBLAS calls use the primary stream from `StreamController`. Weight gradients and input gradients within the same sublayer are serialized, but different sublayers (FFN vs Attention) could potentially overlap with proper stream management (not currently implemented).

---

*Document generated for GRIM-text v2.0 backward pass implementation.*
