# GRIM-text Forward Pass Architecture

## Executive Summary

The GRIM-text forward pass implements a **3-phase GPU-accelerated pipeline** that transforms tokenized input sequences into vocabulary logits. This document details the complete data flow from token IDs through embedding, encoder layers, and the final LM head projection.

**Key Characteristics:**
- **3-Phase Architecture**: Phase 3 (Input) → Phase 2 (Encoder) → Phase 1 (Output)
- **Flash Attention v2**: Memory-efficient attention with O(N) complexity
- **GQA (Grouped Query Attention)**: 3:1 Q-to-KV head ratio for memory efficiency
- **PBM Hybrid Encoding**: ALiBi + RoPE positional encoding (required, no fallback)
- **Fail-Loud Design**: Rule 20 enforced throughout - crashes on invalid state

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [ForwardContext Structure](#forwardcontext-structure)
3. [Phase 3: Input Layer](#phase-3-input-layer)
4. [Phase 2: Encoder Stack](#phase-2-encoder-stack)
5. [EncodingLayer Pipeline](#encodinglayer-pipeline)
6. [Phase 1: Output Layer](#phase-1-output-layer)
7. [Flash Attention v2 Integration](#flash-attention-v2-integration)
8. [Positional Encoding (PBM)](#positional-encoding-pbm)
9. [Activation Caching](#activation-caching)
10. [Error Handling](#error-handling)
11. [Performance Characteristics](#performance-characteristics)
12. [File Reference](#file-reference)

---

## Architecture Overview

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FORWARD PASS PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Token IDs        Phase 3: Input Layer                                      │
│  [B × S]     ──────────────────────────────────────────────────────────►    │
│                   │                                                         │
│                   ▼                                                         │
│              ┌─────────────┐                                                │
│              │   Upload    │  H2D copy: tokens → GPU                        │
│              │   Tokens    │                                                │
│              └──────┬──────┘                                                │
│                     │                                                       │
│                     ▼                                                       │
│              ┌─────────────┐                                                │
│              │  Embedding  │  Token lookup + Atom injection                 │
│              │   Lookup    │  [B×S] → [B×S×d_model]                        │
│              └──────┬──────┘                                                │
│                     │                                                       │
│                     ▼                                                       │
│              ┌─────────────┐                                                │
│              │ ScratchBlock│  Structural atom embeddings                    │
│              │  Forward    │  (optional)                                    │
│              └──────┬──────┘                                                │
│                     │                                                       │
├─────────────────────┼───────────────────────────────────────────────────────┤
│                     │                                                       │
│  Hidden States  Phase 2: Encoder Stack                                      │
│  [B×S×768]   ◄──────┴──────────────────────────────────────────────────►    │
│                   │                                                         │
│                   ▼                                                         │
│         ┌─────────────────────────────────────────────────┐                 │
│         │              ENCODER LAYER × 12                 │                 │
│         │  ┌─────────┐  ┌──────┐  ┌───────┐  ┌─────────┐ │                 │
│         │  │ RMSNorm1│→│ QKV  │→│FlashAttn│→│ RMSNorm2│ │                 │
│         │  └─────────┘  └──────┘  └───────┘  └─────────┘ │                 │
│         │       │           │          │          │      │                 │
│         │       ▼           ▼          ▼          ▼      │                 │
│         │  ┌─────────┐  ┌──────┐  ┌───────┐  ┌─────────┐ │                 │
│         │  │Residual1│  │ RoPE │  │  W_o  │  │   FFN   │ │                 │
│         │  └─────────┘  └──────┘  └───────┘  └─────────┘ │                 │
│         │                                         │      │                 │
│         │                                   ┌─────┴────┐ │                 │
│         │                                   │Residual2 │ │                 │
│         │                                   └──────────┘ │                 │
│         └────────────────────────────────────────────────┘                 │
│                     │                                                       │
├─────────────────────┼───────────────────────────────────────────────────────┤
│                     │                                                       │
│  Encoder Output  Phase 1: Output Layer                                      │
│  [B×S×768]   ◄──────┴──────────────────────────────────────────────────►    │
│                   │                                                         │
│                   ▼                                                         │
│              ┌─────────────┐                                                │
│              │   LM Head   │  [B×S×768] @ W_vocab^T → [B×S×vocab]           │
│              │ Projection  │  Tied weights with embedding                  │
│              └──────┬──────┘                                                │
│                     │                                                       │
│                     ▼                                                       │
│              ┌─────────────┐                                                │
│              │   Logits    │  [B × S × 37,555]                              │
│              │   Output    │                                                │
│              └─────────────┘                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 37,555 | Total vocabulary including atoms |
| `d_model` | 768 | Hidden dimension |
| `num_layers` | 12 | Encoder layers |
| `num_heads` | 12 | Query heads |
| `num_kv_heads` | 4 | Key/Value heads (GQA 3:1) |
| `head_dim` | 64 | Per-head dimension (768/12) |
| `d_ff` | 3,072 | FFN intermediate dimension (4×d_model) |

---

## ForwardContext Structure

The `ForwardContext` struct serves as the central state container passed through all forward phases.

### Definition

```cpp
// File: Layers/ForwardOps/ForwardContext.hpp

struct ForwardContext {
    // ═══════════════════════════════════════════════════════════════════
    // Configuration and Model References
    // ═══════════════════════════════════════════════════════════════════
    const ModelConfig* config;              // Model hyperparameters
    TrainingState_GPU* training_state;      // GPU buffers and optimizer state
    LanguageModel* model;                   // Full language model reference
    
    // ═══════════════════════════════════════════════════════════════════
    // Batch Dimensions
    // ═══════════════════════════════════════════════════════════════════
    int batch_size;                         // Current batch size (B)
    int seq_len;                            // Sequence length (S)
    int total_tokens;                       // B × S
    
    // ═══════════════════════════════════════════════════════════════════
    // Forward Mode Configuration
    // ═══════════════════════════════════════════════════════════════════
    ForwardMode mode;                       // TrainingFull, Prefill, DecodeFull, DecodeIncremental
    ForwardLogitsTarget logits_target;      // FullSequence or LastToken
    
    // ═══════════════════════════════════════════════════════════════════
    // GPU Buffers (Input/Output)
    // ═══════════════════════════════════════════════════════════════════
    const int32_t* d_token_ids;             // Input tokens [B×S] on GPU
    float* d_encoder_output;                // Encoder final output [B×S×d_model]
    float* d_logits;                        // Output logits [B×S×vocab] or [B×vocab]
    
    // ═══════════════════════════════════════════════════════════════════
    // Activation Caches (for backward pass)
    // ═══════════════════════════════════════════════════════════════════
    float* d_embeddings_cache;              // Cached embeddings [B×S×d_model]
    std::vector<LayerCache>* layer_caches;  // Per-layer activation caches
    
    // ═══════════════════════════════════════════════════════════════════
    // Status Tracking
    // ═══════════════════════════════════════════════════════════════════
    ForwardStatus status;                   // Current status code
    std::string error_message;              // Error description if failed
    
    // ═══════════════════════════════════════════════════════════════════
    // CUDA Resources
    // ═══════════════════════════════════════════════════════════════════
    cudaStream_t stream;                    // Primary CUDA stream (REQUIRED)
    cublasHandle_t cublas_handle;           // cuBLAS handle (REQUIRED)
};
```

### ForwardMode Enum

```cpp
enum class ForwardMode {
    TrainingFull,      // Full sequence forward for training (caches activations)
    Prefill,           // Initial prompt processing (KV cache population)
    DecodeFull,        // Full inference without KV cache
    DecodeIncremental  // Single-token generation with KV cache
};
```

### ForwardStatus Enum

```cpp
enum class ForwardStatus {
    SUCCESS = 0,       // Forward pass completed successfully
    INVALID_STATE,     // Context not properly initialized
    CUDA_ERROR,        // CUDA kernel or memory operation failed
    CUBLAS_ERROR,      // cuBLAS matrix operation failed
    NULL_POINTER       // Required pointer was NULL (Rule 20 violation)
};
```

---

## Phase 3: Input Layer

Phase 3 transforms token IDs into dense embeddings, optionally injecting structural atom embeddings from the ScratchBlock system.

### File: `ForwardPhase3_InputLayer.cu`

### Processing Steps

```
Token IDs [B×S]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Token Upload (if needed)                                    │
│   cudaMemcpyAsync(d_tokens, h_tokens, B×S×sizeof(int32_t), H2D)    │
│   - Skip if tokens already on GPU                                   │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Embedding Runtime Forward                                   │
│   embeddingRuntimeForward(ctx, d_tokens, d_embeddings, positions)   │
│   - Token lookup from embedding matrix [vocab × d_model]            │
│   - Position encoding applied (learned positional embeddings)       │
│   - Output: [B×S×d_model] float tensor                             │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Activation Quantization (Optional)                          │
│   if (config.quantize_embeddings) {                                 │
│       quantizeActivations(d_embeddings, quant_params);              │
│   }                                                                 │
│   - Int8 symmetric quantization for inference speedup               │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: ScratchBlock Forward (if enabled)                           │
│   if (config.scratch_block_reasoning) {                             │
│       scratchBlockForward(ctx, d_embeddings, atom_mask);            │
│   }                                                                 │
│   - Injects structural embeddings for detected atoms                │
│   - Numbers, URLs, emails, paths, dates, code literals              │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
Dense Embeddings [B×S×768]
```

### Code Implementation

```cpp
ForwardStatus executePhase3_InputLayer(ForwardContext& ctx) {
    FWD_CHECK_PTR(ctx.d_token_ids, "d_token_ids");
    FWD_CHECK_PTR(ctx.training_state, "training_state");
    
    // Rule 20: Stream MUST be provided - no fallback to default stream
    if (!ctx.stream) {
        return FWD_FAIL_LOUD(ForwardStatus::NULL_POINTER, 
                             "ctx.stream is NULL - caller MUST provide valid stream");
    }
    
    const auto& cfg = *ctx.config;
    const int total_tokens = ctx.batch_size * ctx.seq_len;
    
    // Step 1: Upload tokens to GPU if needed (already on GPU for training)
    // Step 2: Embedding lookup
    embeddingRuntimeForward(
        ctx,
        ctx.d_token_ids,
        ctx.d_embeddings_cache,
        nullptr  // positions computed internally as idx % seq_len
    );
    
    // Step 3: Optional quantization
    if (cfg.quantize_embeddings) {
        quantizeActivations(ctx.d_embeddings_cache, total_tokens * cfg.d_model,
                           ctx.training_state->embedding_quant_scale, ctx.stream);
    }
    
    // Step 4: ScratchBlock atom injection
    if (cfg.scratch_block_reasoning && ctx.model->getScratchBlock()) {
        ctx.model->getScratchBlock()->forward(
            ctx.d_embeddings_cache,
            ctx.d_token_ids,
            total_tokens,
            ctx.stream
        );
    }
    
    return ForwardStatus::SUCCESS;
}
```

---

## Phase 2: Encoder Stack

Phase 2 processes embeddings through 12 transformer encoder layers, using Flash Attention v2 with GQA and RoPE positional encoding.

### File: `ForwardPhase2_Encoder.cu`

### Two Execution Paths

| Mode | Function | Use Case |
|------|----------|----------|
| **Full Sequence** | `runFullEncoder()` | Training, prefill, full inference |
| **Incremental** | `runIncrementalEncoder()` | Single-token decode with KV cache |

### Full Encoder Flow

```
Embeddings [B×S×768]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FOR each layer (0..11):                                             │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ EncodingLayer::forward(input, output, layer_cache)          │   │
│   │                                                             │   │
│   │ See detailed 10-step pipeline below                         │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Optional: Per-layer activation quantization                       │
│   Optional: Attention entropy logging                               │
│                                                                     │
│   layer_output becomes next layer's input                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
Final Encoder Output [B×S×768]
```

### Code: Full Encoder

```cpp
ForwardStatus runFullEncoder(ForwardContext& ctx) {
    FWD_CHECK_PTR(ctx.training_state->encoder, "encoder");
    
    // Rule 20: PBM (positional encoding) MUST be initialized
    if (!ctx.training_state->pbm) {
        return FWD_FAIL_LOUD(ForwardStatus::NULL_POINTER,
                             "PBM (positional encoding) is NULL - MUST initialize before forward");
    }
    
    const auto& cfg = *ctx.config;
    const int n_layers = cfg.num_layers;
    
    float* current_input = ctx.d_embeddings_cache;
    float* current_output = ctx.training_state->encoder_scratch_a;
    
    for (int layer = 0; layer < n_layers; ++layer) {
        auto& layer_cache = (*ctx.layer_caches)[layer];
        
        // Build forward args for this layer
        EncodingForwardArgs args{};
        args.input = current_input;
        args.output = current_output;
        args.batch_size = ctx.batch_size;
        args.seq_len = ctx.seq_len;
        args.pbm = ctx.training_state->pbm;  // REQUIRED for RoPE
        args.stream = ctx.stream;
        
        // Cache pointers for backward pass (training mode)
        if (ctx.mode == ForwardMode::TrainingFull) {
            args.cache_q = layer_cache.q;
            args.cache_k = layer_cache.k;
            args.cache_v = layer_cache.v;
            args.cache_softmax_lse = layer_cache.softmax_lse;
            args.cache_attn_output = layer_cache.attn_output;
            args.cache_residual1 = layer_cache.residual1;
            args.cache_ln1_output = layer_cache.ln1_output;
            args.cache_ln2_output = layer_cache.ln2_output;
            args.cache_ffn_output = layer_cache.ffn_output;
        }
        
        // Execute layer forward
        ctx.training_state->encoder->getLayer(layer).forward(args, nullptr);
        
        // Swap buffers
        std::swap(current_input, current_output);
    }
    
    // Final output in current_input after odd/even swap
    ctx.d_encoder_output = current_input;
    
    return ForwardStatus::SUCCESS;
}
```

### Incremental Decoder Flow

For autoregressive generation, only the new token is processed while using cached KV states:

```cpp
ForwardStatus runIncrementalEncoder(ForwardContext& ctx) {
    // Only process single new token (seq_len = 1)
    assert(ctx.seq_len == 1);
    
    for (int layer = 0; layer < ctx.config->num_layers; ++layer) {
        auto& kv_cache = ctx.training_state->kv_caches[layer];
        
        // Append new K/V to cache
        computeQKV_Incremental(ctx, layer, kv_cache);
        
        // Attention over full cached sequence
        computeIncrementalAttention(ctx, layer, kv_cache);
        
        // FFN on single token
        computeFFN_Incremental(ctx, layer);
    }
    
    return ForwardStatus::SUCCESS;
}
```

---

## EncodingLayer Pipeline

Each encoder layer executes a **10-step forward pipeline** following the pre-norm transformer architecture.

### File: `Layers/Encoding/Encoding_GPU.cu`

### 10-Step Pipeline Diagram

```
Input [T×768]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: RMSNorm1                                                    │
│   ln1_out = RMSNorm(input, gamma_1)                                │
│   - Pre-attention normalization                                     │
│   - gamma: learned scale [768]                                      │
│   - eps: 1e-6                                                       │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: QKV Projection                                              │
│   qkv = ln1_out @ W_qkv^T                                          │
│   - Single fused projection for efficiency                          │
│   - W_qkv shape: [1280 × 768] (GQA-aware)                          │
│   - Split into Q[12 heads], K[4 heads], V[4 heads]                 │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Reshape to BHSD                                             │
│   Q: [B×S×768] → [B, 12, S, 64]   (12 query heads)                 │
│   K: [B×S×256] → [B, 4, S, 64]    (4 KV heads)                     │
│   V: [B×S×256] → [B, 4, S, 64]    (4 KV heads)                     │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3b: RoPE Application                                           │
│   Q_rotated = apply_rope(Q, freqs_cis)                             │
│   K_rotated = apply_rope(K, freqs_cis)                             │
│   - Rotary Position Embedding for relative position encoding        │
│   - Applied per-head independently                                  │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Flash Attention v2                                          │
│   attn_out = FlashAttention(Q, K, V, causal_mask, alibi_slopes)    │
│   - BF16 computation path                                           │
│   - Causal masking (upper triangle)                                 │
│   - ALiBi slopes for distance bias                                  │
│   - Returns: [B, 12, S, 64] attention output                       │
│   - Side output: softmax_lse for backward                           │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Reshape to [T × d_model]                                    │
│   attn_flat: [B, 12, S, 64] → [B×S, 768]                           │
│   - Concatenate all heads                                           │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Output Projection                                           │
│   proj_out = attn_flat @ W_o^T + b_o                               │
│   - W_o shape: [768 × 768]                                         │
│   - Projects concatenated heads back to d_model                     │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 7: Residual Connection 1                                       │
│   residual1 = input + proj_out                                     │
│   - Skip connection around attention block                          │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 8: RMSNorm2                                                    │
│   ln2_out = RMSNorm(residual1, gamma_2)                            │
│   - Pre-FFN normalization                                           │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 9: Feed-Forward Network (FFN)                                  │
│   hidden = GELU(ln2_out @ W1^T + b1)    [T × 3072]                 │
│   ffn_out = hidden @ W2^T + b2          [T × 768]                  │
│   - Two-layer MLP with GELU activation                              │
│   - Expansion factor: 4× (768 → 3072 → 768)                        │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 10: Residual Connection 2                                      │
│   output = residual1 + ffn_out                                     │
│   - Skip connection around FFN block                                │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
Output [T×768] → Next Layer
```

### Code Implementation

```cpp
void EncodingLayer::forward(const EncodingForwardArgs& args, LayerWorkspace<float>* ws) {
    const int total_tokens = args.batch_size * args.seq_len;
    const cudaStream_t stream = args.stream;
    
    // ════════════════════════════════════════════════════════════════════
    // Step 1: RMSNorm1 (Pre-Attention)
    // ════════════════════════════════════════════════════════════════════
    float* ln1_out = args.cache_ln1_output ? args.cache_ln1_output : scratch_ln1_.ptr();
    launchRMSNorm(args.input, ln1_out, rms_gamma1_.ptr(),
                  total_tokens, config_.d_model, config_.rms_eps, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 2: QKV Projection (Fused, GQA-aware)
    // ════════════════════════════════════════════════════════════════════
    const int head_dim = config_.d_model / config_.num_heads;
    const int kv_dim = config_.num_kv_heads * head_dim;
    const int total_qkv_dim = config_.d_model + 2 * kv_dim;  // Q + K + V
    
    float* qkv_flat = scratch_qkv_.ptr();
    const float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(config_.cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                total_qkv_dim, total_tokens, config_.d_model,
                &alpha,
                W_qkv_.ptr(), config_.d_model,
                ln1_out, config_.d_model,
                &beta,
                qkv_flat, total_qkv_dim);
    
    // Split Q, K, V from fused output
    float* Q_flat = qkv_flat;                              // [T × d_model]
    float* K_flat = qkv_flat + config_.d_model;            // [T × kv_dim]
    float* V_flat = qkv_flat + config_.d_model + kv_dim;   // [T × kv_dim]
    
    // ════════════════════════════════════════════════════════════════════
    // Step 3: Reshape [T, dim] → [B, H, S, head_dim] (BHSD format)
    // ════════════════════════════════════════════════════════════════════
    float* Q_bhsd = args.cache_q ? args.cache_q : scratch_q_bhsd_.ptr();
    float* K_bhsd = args.cache_k ? args.cache_k : scratch_k_bhsd_.ptr();
    float* V_bhsd = args.cache_v ? args.cache_v : scratch_v_bhsd_.ptr();
    
    TensorContract::reshape_to_bhsd(Q_flat, Q_bhsd, args.batch_size, args.seq_len,
                                    config_.num_heads, head_dim, stream);
    TensorContract::reshape_to_bhsd(K_flat, K_bhsd, args.batch_size, args.seq_len,
                                    config_.num_kv_heads, head_dim, stream);
    TensorContract::reshape_to_bhsd(V_flat, V_bhsd, args.batch_size, args.seq_len,
                                    config_.num_kv_heads, head_dim, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 3b: RoPE Application (Rotary Position Embedding)
    // ════════════════════════════════════════════════════════════════════
    // Rule 20: PBM MUST be provided - no fallback
    if (!args.pbm) {
        throw std::runtime_error("EncodingLayer::forward: args.pbm is NULL - "
                                 "PBM (positional encoding) MUST be initialized");
    }
    
    args.pbm->applyRoPE(Q_bhsd, args.batch_size, args.seq_len, config_.num_heads, head_dim, stream);
    args.pbm->applyRoPE(K_bhsd, args.batch_size, args.seq_len, config_.num_kv_heads, head_dim, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 4: Flash Attention v2
    // ════════════════════════════════════════════════════════════════════
    float* attn_out_bhsd = scratch_attn_bhsd_.ptr();
    float* softmax_lse = args.cache_softmax_lse ? args.cache_softmax_lse : scratch_lse_.ptr();
    
    // Get ALiBi slopes from PBM
    const float* alibi_slopes = args.pbm->getALiBiSlopes();
    
    // Launch Flash Attention v2 (BF16 path)
    launchFlashAttentionForward(
        Q_bhsd, K_bhsd, V_bhsd,           // Input Q, K, V
        attn_out_bhsd,                     // Output
        softmax_lse,                       // Side output for backward
        args.batch_size,
        args.seq_len,                      // seqlen_q
        args.seq_len,                      // seqlen_k (same for self-attention)
        config_.num_heads,                 // h (query heads)
        config_.num_kv_heads,              // h_k (KV heads, GQA)
        head_dim,
        1.0f / sqrtf(static_cast<float>(head_dim)),  // softmax scale
        true,                              // is_causal
        alibi_slopes,                      // ALiBi attention bias
        stream
    );
    
    // ════════════════════════════════════════════════════════════════════
    // Step 5: Reshape BHSD → [T, d_model]
    // ════════════════════════════════════════════════════════════════════
    float* attn_flat = args.cache_attn_output ? args.cache_attn_output : scratch_attn_flat_.ptr();
    TensorContract::reshape_from_bhsd(attn_out_bhsd, attn_flat, args.batch_size, args.seq_len,
                                      config_.num_heads, head_dim, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 6: Output Projection (W_o)
    // ════════════════════════════════════════════════════════════════════
    float* proj_out = scratch_proj_.ptr();
    
    cublasSgemm(config_.cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                config_.d_model, total_tokens, config_.d_model,
                &alpha,
                W_o_.ptr(), config_.d_model,
                attn_flat, config_.d_model,
                &beta,
                proj_out, config_.d_model);
    
    if (config_.use_bias && b_o_.ptr()) {
        addBias(proj_out, b_o_.ptr(), config_.d_model, total_tokens, stream);
    }
    
    // ════════════════════════════════════════════════════════════════════
    // Step 7: Residual Connection 1
    // ════════════════════════════════════════════════════════════════════
    float* residual1 = args.cache_residual1 ? args.cache_residual1 : scratch_residual1_.ptr();
    launchResidualAdd(args.input, proj_out, residual1, total_tokens * config_.d_model, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 8: RMSNorm2 (Pre-FFN)
    // ════════════════════════════════════════════════════════════════════
    float* ln2_out = args.cache_ln2_output ? args.cache_ln2_output : scratch_ln2_.ptr();
    launchRMSNorm(residual1, ln2_out, rms_gamma2_.ptr(),
                  total_tokens, config_.d_model, config_.rms_eps, stream);
    
    // ════════════════════════════════════════════════════════════════════
    // Step 9: Feed-Forward Network
    // ════════════════════════════════════════════════════════════════════
    float* ffn_out = scratch_ffn_.ptr();
    float* pre_gelu = scratch_pre_gelu_.ptr();
    float* post_gelu = args.cache_ffn_output ? args.cache_ffn_output : scratch_post_gelu_.ptr();
    
    FeedForwardForwardArgs ffn_args{};
    ffn_args.input = ln2_out;
    ffn_args.output = ffn_out;
    ffn_args.total_tokens = total_tokens;
    ffn_args.cache_pre_gelu = pre_gelu;
    ffn_args.cache_post_gelu = post_gelu;
    
    ffn_->forward(ffn_args, nullptr);
    
    // CRITICAL: Cache post-GELU for backward pass (Issue #25 fix)
    if (args.cache_ffn_output) {
        cudaMemcpyAsync(args.cache_ffn_output, post_gelu,
                        static_cast<size_t>(total_tokens) * config_.d_ff * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    
    // ════════════════════════════════════════════════════════════════════
    // Step 10: Residual Connection 2
    // ════════════════════════════════════════════════════════════════════
    launchResidualAdd(residual1, ffn_out, args.output, total_tokens * config_.d_model, stream);
}
```

---

## Phase 1: Output Layer

Phase 1 projects encoder outputs to vocabulary logits using the LM head (weight-tied with embeddings).

### File: `ForwardPhase1_OutputLayer.cu`

### Processing Steps

```
Encoder Output [B×S×768]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Select Token Range                                          │
│   if (logits_target == LastToken) {                                │
│       input = encoder_output[:, -1, :]  // [B × 768]               │
│       output_tokens = B                                             │
│   } else {                                                          │
│       input = encoder_output            // [B×S × 768]             │
│       output_tokens = B × S                                         │
│   }                                                                 │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: LM Head Projection                                          │
│   logits = input @ W_vocab^T                                       │
│   - W_vocab: [vocab_size × d_model] = [37555 × 768]                │
│   - Tied with embedding matrix (same pointer)                       │
│   - cuBLAS SGEMM: [T × 768] @ [768 × 37555] → [T × 37555]         │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Numeric Head (Optional)                                     │
│   if (config.numeric_head_enabled) {                               │
│       numeric_out = input @ W_numeric^T                            │
│       // Predicts numeric values for detected number atoms          │
│   }                                                                 │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Logits Quantization (Optional)                              │
│   if (config.quantize_logits) {                                    │
│       quantizeActivations(logits, quant_params);                   │
│   }                                                                 │
└────────────────────────────────────────────────────────────────────┬┘
      │
      ▼
Logits [B×S×37555] or [B×37555]
```

### Code Implementation

```cpp
ForwardStatus executePhase1_OutputLayer(ForwardContext& ctx) {
    FWD_CHECK_PTR(ctx.d_encoder_output, "d_encoder_output");
    FWD_CHECK_PTR(ctx.d_logits, "d_logits");
    FWD_CHECK_PTR(ctx.model, "model");
    
    const auto& cfg = *ctx.config;
    
    // Determine output dimensions based on logits target
    int output_tokens;
    float* lm_head_input;
    
    if (ctx.logits_target == ForwardLogitsTarget::LastToken) {
        // Only compute logits for last token (inference)
        output_tokens = ctx.batch_size;
        lm_head_input = ctx.d_encoder_output + 
                       (ctx.seq_len - 1) * cfg.d_model;  // Offset to last token
    } else {
        // Full sequence logits (training)
        output_tokens = ctx.total_tokens;
        lm_head_input = ctx.d_encoder_output;
    }
    
    // LM Head forward projection
    LMHeadForwardParams lm_params{};
    lm_params.encoder_output = lm_head_input;
    lm_params.weights = ctx.model->getLMHeadWeights();  // Tied with embeddings
    lm_params.bias = cfg.use_bias ? ctx.model->getLMHeadBias() : nullptr;
    lm_params.logits = ctx.d_logits;
    lm_params.batch_size = (ctx.logits_target == ForwardLogitsTarget::LastToken) 
                           ? ctx.batch_size : ctx.batch_size;
    lm_params.seq_len = (ctx.logits_target == ForwardLogitsTarget::LastToken)
                        ? 1 : ctx.seq_len;
    lm_params.d_model = cfg.d_model;
    lm_params.vocab_size = cfg.vocab_size;
    lm_params.handle = ctx.cublas_handle;
    lm_params.stream = ctx.stream;
    lm_params.use_bias = cfg.use_bias;
    
    launchLMHeadForward(lm_params);
    
    // Optional: Numeric head for number prediction
    if (cfg.numeric_head_enabled && ctx.model->getNumericHead()) {
        ctx.model->getNumericHead()->forward(
            lm_head_input,
            ctx.d_numeric_output,
            output_tokens,
            ctx.stream
        );
    }
    
    // Optional: Logits quantization
    if (cfg.quantize_logits) {
        quantizeActivations(ctx.d_logits, output_tokens * cfg.vocab_size,
                           ctx.training_state->logits_quant_scale, ctx.stream);
    }
    
    return ForwardStatus::SUCCESS;
}
```

---

## Flash Attention v2 Integration

### File: `Layers/FlashAttention/Flash_Attention_Kernal.cu`

Flash Attention v2 provides memory-efficient attention computation with O(N) memory complexity vs O(N²) for standard attention.

### Key Features

| Feature | Description |
|---------|-------------|
| **Memory Efficiency** | O(N) memory via tiled computation |
| **BF16 Computation** | Brain floating-point for training stability |
| **GQA Support** | Grouped Query Attention (3:1 ratio) |
| **Causal Masking** | Upper triangle masked for autoregressive |
| **ALiBi Slopes** | Linear attention bias for position encoding |
| **Softmax LSE Output** | Log-sum-exp for backward pass |

### Parameter Structure

```cpp
struct Flash_fwd_params : public Qkv_params {
    // Output pointers
    void* __restrict__ o_ptr;           // Attention output
    void* __restrict__ softmax_lse_ptr; // Log-sum-exp for backward
    
    // Dimensions
    int b;              // Batch size
    int seqlen_q;       // Query sequence length
    int seqlen_k;       // Key sequence length
    int d;              // Head dimension
    int h;              // Query heads
    int h_k;            // KV heads (GQA)
    int h_h_k_ratio;    // h / h_k = heads_per_kv_group
    
    // Scaling
    float scale_softmax;      // 1/sqrt(d)
    float scale_softmax_log2; // log2(scale_softmax)
    
    // Strides (BHSD layout)
    index_t q_batch_stride, q_row_stride, q_head_stride;
    index_t k_batch_stride, k_row_stride, k_head_stride;
    index_t v_batch_stride, v_row_stride, v_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride;
    
    // Optional features
    float* __restrict__ alibi_slopes_ptr;  // ALiBi position bias
    bool is_causal;                        // Causal masking flag
};
```

### Launch Configuration

```cpp
void launchFlashAttentionForward(
    const float* Q, const float* K, const float* V,
    float* O, float* softmax_lse,
    int batch_size, int seqlen_q, int seqlen_k,
    int num_heads, int num_kv_heads, int head_dim,
    float softmax_scale, bool is_causal,
    const float* alibi_slopes, cudaStream_t stream)
{
    // Convert FP32 to BF16 for kernel
    __nv_bfloat16* Q_bf16 = convertToBF16(Q, ...);
    __nv_bfloat16* K_bf16 = convertToBF16(K, ...);
    __nv_bfloat16* V_bf16 = convertToBF16(V, ...);
    
    Flash_fwd_params params;
    params.q_ptr = Q_bf16;
    params.k_ptr = K_bf16;
    params.v_ptr = V_bf16;
    params.o_ptr = O_bf16;
    params.softmax_lse_ptr = softmax_lse;
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_kv_heads;
    params.h_h_k_ratio = num_heads / num_kv_heads;  // GQA ratio
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = head_dim;
    params.scale_softmax = softmax_scale;
    params.is_causal = is_causal;
    params.alibi_slopes_ptr = alibi_slopes;
    
    // Set strides for BHSD layout
    params.q_head_stride = head_dim;
    params.q_row_stride = num_heads * head_dim;
    params.q_batch_stride = seqlen_q * num_heads * head_dim;
    // ... similar for K, V, O
    
    // Launch kernel (SM86 optimized)
    run_flash_fwd(params, stream);
    
    // Convert BF16 output back to FP32
    convertToFP32(O_bf16, O, ...);
}
```

---

## Positional Encoding (PBM)

PBM (Positional Bias Module) implements a hybrid ALiBi + RoPE positional encoding system.

### Components

| Component | Purpose |
|-----------|---------|
| **RoPE** | Rotary Position Embedding for relative positions |
| **ALiBi** | Attention Linear Bias for distance-based attention decay |

### RoPE Application

```cpp
void PBM::applyRoPE(float* tensor, int batch_size, int seq_len, 
                    int num_heads, int head_dim, cudaStream_t stream) {
    // tensor shape: [B, H, S, head_dim]
    // RoPE applied in-place to each head
    
    // Precomputed frequency pairs
    // freqs[i] = base^(-2i/head_dim) for i in [0, head_dim/2)
    
    launchRoPEKernel(
        tensor,
        freqs_cos_.ptr(),  // cos(pos * freqs)
        freqs_sin_.ptr(),  // sin(pos * freqs)
        batch_size, seq_len, num_heads, head_dim,
        stream
    );
}
```

### ALiBi Slopes

```cpp
void PBM::initALiBiSlopes(int num_heads) {
    // ALiBi slopes: geometric sequence
    // slope[h] = 2^(-8 * (h+1) / num_heads)
    // For num_heads=12: slopes from ~0.0039 to ~0.5
    
    std::vector<float> slopes(num_heads);
    const float base = std::pow(2.0f, -8.0f / num_heads);
    for (int h = 0; h < num_heads; ++h) {
        slopes[h] = std::pow(base, h + 1);
    }
    
    // Upload to GPU
    alibi_slopes_.allocate(num_heads);
    cudaMemcpy(alibi_slopes_.ptr(), slopes.data(), 
               num_heads * sizeof(float), cudaMemcpyHostToDevice);
}
```

---

## Activation Caching

For training, activations are cached at each layer for the backward pass.

### LayerCache Structure

```cpp
struct LayerCache {
    // Attention caches
    float* q;              // Query [B, H, S, head_dim]
    float* k;              // Key [B, H_kv, S, head_dim]
    float* v;              // Value [B, H_kv, S, head_dim]
    float* softmax_lse;    // Log-sum-exp [B, H, S]
    float* attn_output;    // Attention output [T, d_model]
    
    // Normalization caches
    float* ln1_output;     // Pre-attention norm output
    float* ln2_output;     // Pre-FFN norm output
    
    // Residual caches
    float* residual1;      // After attention residual
    
    // FFN caches
    float* ffn_output;     // Post-GELU (CRITICAL for grad_W2)
};
```

### Memory Requirements

| Cache Component | Size per Layer | Total (12 layers) |
|-----------------|----------------|-------------------|
| Q cache | B × S × 768 | B × S × 9,216 |
| K cache | B × S × 256 | B × S × 3,072 |
| V cache | B × S × 256 | B × S × 3,072 |
| softmax_lse | B × 12 × S | B × S × 144 |
| attn_output | B × S × 768 | B × S × 9,216 |
| ln1_output | B × S × 768 | B × S × 9,216 |
| ln2_output | B × S × 768 | B × S × 9,216 |
| residual1 | B × S × 768 | B × S × 9,216 |
| ffn_output | B × S × 3072 | B × S × 36,864 |

**Example**: For B=3, S=720:
- Per layer: ~22 MB
- Total 12 layers: ~264 MB

---

## Error Handling

The forward pass enforces **Rule 20: Fail Loud** - crashes immediately on invalid state.

### Macros

```cpp
// Fail with detailed error message
#define FWD_FAIL_LOUD(status, msg) \
    do { \
        fprintf(stderr, "[FATAL] %s at %s:%d\n", msg, __FILE__, __LINE__); \
        return status; \
    } while(0)

// Check pointer is not NULL
#define FWD_CHECK_PTR(ptr, name) \
    do { \
        if (!(ptr)) { \
            return FWD_FAIL_LOUD(ForwardStatus::NULL_POINTER, \
                                 name " is NULL - caller MUST provide valid pointer"); \
        } \
    } while(0)

// Check CUDA operation succeeded
#define FWD_CHECK_CUDA(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "CUDA error: " << cudaGetErrorString(err); \
            return FWD_FAIL_LOUD(ForwardStatus::CUDA_ERROR, oss.str().c_str()); \
        } \
    } while(0)

// Check cuBLAS operation succeeded
#define FWD_CHECK_CUBLAS(expr) \
    do { \
        cublasStatus_t status = (expr); \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::ostringstream oss; \
            oss << "cuBLAS error: " << status; \
            return FWD_FAIL_LOUD(ForwardStatus::CUBLAS_ERROR, oss.str().c_str()); \
        } \
    } while(0)
```

### Critical Validation Points

| Check | Location | Consequence |
|-------|----------|-------------|
| `ctx.stream != nullptr` | Phase 3 entry | Crash (Rule 20) |
| `args.pbm != nullptr` | EncodingLayer | Crash (Rule 20) |
| `ctx.cublas_handle != nullptr` | All phases | Crash (Rule 20) |
| `d_token_ids != nullptr` | Phase 3 | Crash (Rule 20) |
| `total_tokens > 0` | All phases | Crash (Rule 20) |

---

## Performance Characteristics

### Per-Phase Timing (Typical)

| Phase | Operation | Time (ms) | % Total |
|-------|-----------|-----------|---------|
| Phase 3 | Token upload | 0.02 | <1% |
| Phase 3 | Embedding lookup | 0.15 | 2% |
| Phase 3 | ScratchBlock | 0.05 | <1% |
| Phase 2 | RMSNorm (×24) | 0.48 | 6% |
| Phase 2 | QKV projection (×12) | 1.20 | 15% |
| Phase 2 | Flash Attention (×12) | 3.60 | 45% |
| Phase 2 | Output projection (×12) | 0.60 | 8% |
| Phase 2 | FFN (×12) | 1.44 | 18% |
| Phase 1 | LM Head | 0.50 | 6% |
| **Total** | | **~8.0** | 100% |

*Measured on RTX 3080Ti, B=3, S=720, d_model=768*

### Memory Layout

All tensors use **row-major** layout with these conventions:

| Tensor | Shape | Stride Order |
|--------|-------|--------------|
| Embeddings | [T, d_model] | Row-major [d_model, 1] |
| Q/K/V (BHSD) | [B, H, S, head_dim] | [H×S×hd, S×hd, hd, 1] |
| Attention out | [T, d_model] | Row-major [d_model, 1] |
| Logits | [T, vocab] | Row-major [vocab, 1] |

---

## File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `ForwardOps_Orchestrator.hpp` | Main entry point, function declarations | 50 |
| `ForwardOps_Orchestrator.cu` | Phase coordination, timing | 178 |
| `ForwardContext.hpp` | Context struct, enums, macros | 207 |
| `ForwardPhase3_InputLayer.cu` | Token upload, embedding, ScratchBlock | 180 |
| `ForwardPhase2_Encoder.cu` | Encoder execution (full/incremental) | 423 |
| `ForwardPhase1_OutputLayer.cu` | LM head, numeric head, logits | 140 |
| `Encoding_GPU.cu` | Single layer 10-step forward | 729 |
| `Forward_GPU.cu` | GPUGrimEncoder implementation | 210 |
| `Flash_Attention_Kernal.cu` | Flash Attention v2 forward/backward | 981 |
| `Feed_Forward_GPU.cu` | FFN forward (GELU MLP) | 345 |
| `RMSNorm_Kernel_GPU.cu` | RMSNorm forward/backward | 248 |
| `lm_head_GPU.cu` | LM head projection | 307 |

---

## Appendix: Forward Pass Call Graph

```
executeForward(ctx)                           // ForwardOps_Orchestrator.cu
├── validateForwardContext(ctx)
├── executePhase3_InputLayer(ctx)             // ForwardPhase3_InputLayer.cu
│   ├── embeddingRuntimeForward()
│   ├── quantizeActivations()  [optional]
│   └── scratchBlock->forward() [optional]
├── executePhase2_Encoder(ctx)                // ForwardPhase2_Encoder.cu
│   ├── runFullEncoder()
│   │   └── for layer in 0..11:
│   │       └── EncodingLayer::forward()      // Encoding_GPU.cu
│   │           ├── launchRMSNorm()           // RMSNorm_Kernel_GPU.cu
│   │           ├── cublasSgemm() (QKV)
│   │           ├── TensorContract::reshape_to_bhsd()
│   │           ├── pbm->applyRoPE()
│   │           ├── launchFlashAttentionForward() // Flash_Attention_Kernal.cu
│   │           ├── TensorContract::reshape_from_bhsd()
│   │           ├── cublasSgemm() (W_o)
│   │           ├── launchResidualAdd()
│   │           ├── launchRMSNorm()
│   │           ├── ffn_->forward()           // Feed_Forward_GPU.cu
│   │           └── launchResidualAdd()
│   └── runIncrementalEncoder() [alternative]
└── executePhase1_OutputLayer(ctx)            // ForwardPhase1_OutputLayer.cu
    ├── launchLMHeadForward()                 // lm_head_GPU.cu
    ├── numericHead->forward() [optional]
    └── quantizeActivations() [optional]
```

---

*Document Version: 1.0*  
*Last Updated: December 2025*  
*Author: GRIM Development Team*
