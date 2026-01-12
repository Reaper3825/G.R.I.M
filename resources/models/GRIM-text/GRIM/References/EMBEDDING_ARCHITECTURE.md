# GRIM-text Embedding & Positional Encoding Architecture

**Technical Reference Document**  
**Version:** 1.0  
**Author:** GRIM Development Team  
**Last Updated:** December 2025

---

## Executive Summary

GRIM-text implements a sophisticated embedding system featuring:

1. **Dual Embedding Matrices** — Token embeddings for vocabulary mapping plus learnable positional embeddings
2. **Hybrid Positional Bias Method (PBM)** — Combined ALiBi + RoPE for state-of-the-art position awareness
3. **Weight Tying** — Shared embedding/LM-head matrices for parameter efficiency
4. **GPU-Accelerated Kernels** — Fused embedding+RMSNorm forward pass, atomic scatter-add backward
5. **Xavier/Glorot Initialization** — Proper variance scaling for deep network training stability

This document provides a rigorous mathematical foundation alongside implementation details suitable for academic review.

---

## Table of Contents

1. [Embedding Architecture Overview](#1-embedding-architecture-overview)
2. [Token Embedding Layer](#2-token-embedding-layer)
3. [Positional Encoding System](#3-positional-encoding-system)
4. [Weight Tying & LM Head](#4-weight-tying--lm-head)
5. [Weight Initialization](#5-weight-initialization)
6. [Backward Pass & Gradient Flow](#6-backward-pass--gradient-flow)
7. [GPU Implementation Details](#7-gpu-implementation-details)
8. [Serialization Format](#8-serialization-format)
9. [API Reference](#9-api-reference)
10. [Performance Characteristics](#10-performance-characteristics)

---

## 1. Embedding Architecture Overview

### 1.1 System Design

The GRIM-text embedding pipeline transforms discrete token IDs into continuous dense representations:

```
Token IDs [B × S]  ──┬──► Token Embeddings [B × S × d]  ──┐
                     │                                    │
Position IDs [B × S] ──► Position Embeddings [B × S × d] ─┴─► Sum ──► [Optional RMSNorm] ──► Output [B × S × d]
```

Where:
- `B` = batch size
- `S` = sequence length  
- `d` = model dimension (`d_model`)

### 1.2 Memory Layout Convention

All tensors use **row-major** (C-style) layout:
- Token embeddings: `[vocab_size, d_model]`
- Position embeddings: `[max_position, d_model]`
- Output: `[batch_size × seq_len, d_model]` (flattened for GPU efficiency)

### 1.3 Configuration Structure

```cpp
struct EmbeddingConfig {
    int vocab_size;           // Total vocabulary size (bytes + atoms + unigram)
    int max_position;         // Maximum sequence length for position embeddings
    int d_model;              // Embedding dimension (768 default)
    bool apply_rms_norm;      // Optional post-embedding normalization
    float rms_epsilon;        // RMSNorm numerical stability (1e-5)
    cudaStream_t stream;      // CUDA stream for async operations
};
```

---

## 2. Token Embedding Layer

### 2.1 Mathematical Foundation

The token embedding function maps discrete tokens to continuous vectors:

$$E_{token}: \mathcal{V} \rightarrow \mathbb{R}^d$$

Where $\mathcal{V} = \{0, 1, ..., V-1\}$ is the vocabulary of size $V$.

The embedding lookup is defined as:

$$\mathbf{e}_i = \mathbf{W}_E[t_i, :]$$

Where:
- $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ is the embedding matrix
- $t_i$ is the token ID at position $i$
- $\mathbf{e}_i \in \mathbb{R}^d$ is the resulting embedding vector

### 2.2 Vocabulary Structure

GRIM-text uses a structured vocabulary layout from the tokenizer:

| Range | Token Type | Count | Purpose |
|-------|-----------|-------|---------|
| `[0, 255]` | Byte tokens | 256 | UTF-8 byte fallback for unknown chars |
| `[256, 274]` | Atom placeholders | 19 | Structural element markers (URLs, emails, etc.) |
| `[275, V-1]` | Unigram tokens | V-275 | Learned subword vocabulary |

**Key Constants:**
```cpp
constexpr int BYTE_TOKEN_END = 256;
constexpr int ATOM_TOKEN_START = 256;
constexpr int NUM_ATOM_TYPES = 19;
constexpr int ATOM_TOKEN_END = ATOM_TOKEN_START + NUM_ATOM_TYPES;
```

### 2.3 Token ID Validation

The kernel implements bounds checking with clamping:

```cpp
__device__ inline int clampIndex(int value, int limit) {
    if (limit <= 0) return 0;
    if (value < 0) return 0;
    if (value >= limit) return limit - 1;
    return value;
}
```

Invalid token IDs are silently clamped to valid range. Per Rule 20 ("Fail Loud"), input validation throws exceptions at the API boundary:

```cpp
if (token_id < 0 || token_id >= vocab_size) {
    throw std::runtime_error("Invalid token_id=" + std::to_string(token_id));
}
```

---

## 3. Positional Encoding System

GRIM-text implements a **Hybrid ALiBi + RoPE** positional bias method (PBM), combining the strengths of both approaches.

### 3.1 Position Embedding Lookup

When explicit learnable position embeddings are used:

$$\mathbf{p}_i = \mathbf{W}_P[pos_i, :]$$

Where $\mathbf{W}_P \in \mathbb{R}^{L_{max} \times d}$ and $L_{max}$ is the maximum sequence length.

### 3.2 Auto-Position Computation

When position IDs are not explicitly provided, the kernel computes them automatically:

```cpp
// For batched input with total_tokens = batch_size × seq_len
const int pos_id = positions ? positions[token_idx] : (token_idx % seq_len);
```

This ensures each sequence in a batch gets positions `[0, 1, 2, ..., seq_len-1]`.

### 3.3 Rotary Position Embeddings (RoPE)

RoPE applies position-dependent rotation to query and key vectors in attention:

#### 3.3.1 Inverse Frequency Computation

$$\theta_i = \frac{s}{\Theta^{2i/d_{rot}}}$$

Where:
- $\Theta = 10000$ (base frequency)
- $s$ = scaling factor (1.0 default, adjustable for NTK scaling)
- $d_{rot}$ = rotary dimension (typically equals `head_dim`)
- $i \in [0, d_{rot}/2)$

**Implementation:**
```cpp
for (int i = 0; i < half_dim; ++i) {
    const float exp_arg = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
    inv_freq[i] = scaling / std::pow(theta, exp_arg);
}
```

#### 3.3.2 Rotation Matrix Application

For position $p$ and dimension pair $(i, i+1)$:

$$\begin{pmatrix} x'_i \\ x'_{i+1} \end{pmatrix} = \begin{pmatrix} \cos(\theta_i \cdot p) & -\sin(\theta_i \cdot p) \\ \sin(\theta_i \cdot p) & \cos(\theta_i \cdot p) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+1} \end{pmatrix}$$

**CUDA Implementation:**
```cpp
__device__ __forceinline__ void applyRotation(float& x, float& y, float cos_val, float sin_val) {
    const float x_rot = x * cos_val - y * sin_val;
    const float y_rot = x * sin_val + y * cos_val;
    x = x_rot;
    y = y_rot;
}
```

#### 3.3.3 RoPE Kernel Structure

- **Grid:** `(ceil(seq_len/256), num_heads, batch_size)`
- **Block:** `(256)`
- **Memory Access:** Coalesced BHSD (Batch, Head, Sequence, Dimension) layout

### 3.4 Attention Linear Biases (ALiBi)

ALiBi adds position-dependent bias to attention scores without modifying embeddings:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot |i - j|\right) V$$

#### 3.4.1 Slope Computation

Each attention head $h$ gets a geometric slope:

$$m_h = 2^{\frac{-8 \cdot (h+1)}{H}}$$

Where $H$ is the total number of heads.

**Implementation:**
```cpp
for (int h = 0; h < num_heads; ++h) {
    const float exponent = alibi_slope_exponent * (h + 1) / num_heads;
    slopes[h] = std::pow(2.0f, exponent);
}
```

**Example Slopes (12 heads):**
| Head | Exponent | Slope |
|------|----------|-------|
| 0 | -0.667 | 0.630 |
| 1 | -1.333 | 0.397 |
| 2 | -2.000 | 0.250 |
| ... | ... | ... |
| 11 | -8.000 | 0.004 |

### 3.5 Hybrid ALiBi + RoPE Design

GRIM-text always enables both methods simultaneously:

| Method | Applied At | Target | Effect |
|--------|-----------|--------|--------|
| RoPE | Before attention | Q, K vectors | Relative position via rotation |
| ALiBi | During attention | QK scores | Absolute position via bias |

**Rationale:**
- RoPE encodes **relative** position efficiently
- ALiBi provides **absolute** position awareness
- Combined approach maximizes length generalization

### 3.6 GQA-Aware Positional Encoding

With Grouped Query Attention (GQA), Q and K/V have different head counts:

```cpp
struct PBMConfig {
    int num_heads;      // Q heads (12 default)
    int num_kv_heads;   // K/V heads (4 default, 3:1 ratio)
    // ...
};
```

**GQA RoPE Kernel:**
```cpp
void launchRoPERotationGQA(
    float* Q,                  // [B, num_q_heads, S, head_dim]
    float* K,                  // [B, num_kv_heads, S, head_dim]
    const float* inv_freq,
    int batch_size,
    int num_q_heads,           // 12
    int num_kv_heads,          // 4
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream
);
```

Two separate kernel launches handle different head counts:
1. Q rotation: grid size based on `num_q_heads`
2. K rotation: grid size based on `num_kv_heads`

---

## 4. Weight Tying & LM Head

### 4.1 Weight Tying Principle

When `tie_embeddings = true`, the input embedding matrix $\mathbf{W}_E$ and output projection matrix $\mathbf{W}_{LM}$ share the same memory:

$$\mathbf{W}_{LM} = \mathbf{W}_E^T$$

**Benefits:**
- Parameter count reduced by $V \times d$ (e.g., 37,555 × 768 = 28.8M parameters)
- Improved generalization through input-output consistency
- Reduced memory footprint

### 4.2 Memory Aliasing

```cpp
struct TrainingState {
    // When tie_embeddings=true:
    // embedding_grads == lm_head_weight_grads (same pointer)
    float* embedding_grads;
    float* lm_head_weight_grads;  // Aliased!
    
    // lm_head_weights points to EmbeddingRuntime::token_buffer
    float* lm_head_weights;
    bool lm_head_weights_owned;   // false when tied
};
```

### 4.3 Gradient Accumulation with Tying

Critical implementation detail: gradients from both LM head backward (cuBLAS, overwrites) and embedding backward (atomicAdd) flow to the same buffer.

**Execution Order:**
1. LM head backward: `grad_W_LM = hidden^T @ grad_logits` (cuBLAS, beta=0, overwrites)
2. Embedding backward: `atomicAdd(&grad_W_E[token_id], grad_hidden)` (accumulates)

The atomic operations ensure correct accumulation even with pointer aliasing.

### 4.4 Destructor Safety

```cpp
void TrainingState::freeGradients() {
    const bool grads_are_tied = (embedding_grads == lm_head_weight_grads) && 
                                embedding_grads != nullptr;
    
    if (grads_are_tied) {
        // Free once via lm_head_weight_grads, null out embedding_grads
        if (lm_head_weight_grads) cudaFree(lm_head_weight_grads);
        embedding_grads = nullptr;  // Prevent double-free
        lm_head_weight_grads = nullptr;
    } else {
        // Untied: free both separately
        if (embedding_grads) cudaFree(embedding_grads);
        if (lm_head_weight_grads) cudaFree(lm_head_weight_grads);
    }
}
```

---

## 5. Weight Initialization

### 5.1 Xavier/Glorot Initialization

GRIM-text uses Xavier normal initialization for all weight matrices:

$$\mathbf{W} \sim \mathcal{N}(0, \sigma^2), \quad \sigma = \sqrt{\frac{2}{fan_{in} + fan_{out}}}$$

**Rationale:** Maintains variance of activations and gradients through deep networks, preventing vanishing/exploding gradient problems.

### 5.2 GPU Implementation

```cpp
__global__ void XavierNormalKernel(float* data, size_t elements,
                                   float fan_in, float fan_out,
                                   unsigned long long seed, unsigned long long subseq) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, subseq, &state);

    const float stddev = sqrtf(2.0f / (fan_in + fan_out));
    data[idx] = curand_normal(&state) * stddev;
}
```

**Properties:**
- Uses cuRAND Philox generator (high-quality, parallel-safe)
- Each thread initializes independent RNG state via index
- Sequence offset (`subseq`) ensures different random streams per layer

### 5.3 Layer-Specific Initialization

```cpp
// Attention QKV projection
float xavier_qkv_stddev = sqrt(2.0f / (d_model + total_qkv_dim));
launchXavierInit(w_qkv, qkv_size, xavier_qkv_stddev, seed + layer*4, stream);

// Output projection with GPT-2 residual scaling
float residual_scale = 1.0f / sqrt(2.0f * num_layers);
float xavier_wo_stddev = xavier_qkv_stddev * residual_scale;
launchXavierInit(w_o, wo_size, xavier_wo_stddev, seed + layer*4 + 1, stream);

// FFN W1 (expansion)
float xavier_w1_stddev = sqrt(2.0f / (d_model + d_ff));
launchXavierInit(w1, w1_size, xavier_w1_stddev, seed + layer*4 + 2, stream);

// FFN W2 (projection) with residual scaling
float xavier_w2_stddev = sqrt(2.0f / (d_ff + d_model)) * residual_scale;
launchXavierInit(w2, w2_size, xavier_w2_stddev, seed + layer*4 + 3, stream);
```

**Residual Scaling:** Following GPT-2, output projections are scaled by $1/\sqrt{2N}$ where $N$ is the number of layers. This prevents gradient explosion through residual connections.

### 5.4 Embedding Initialization

Token embeddings use the same Xavier scheme:

```cpp
// Embedding: [vocab_size, d_model]
float xavier_emb_stddev = sqrt(2.0f / (vocab_size + d_model));
launchXavierInit(token_embeddings, vocab_size * d_model, xavier_emb_stddev, seed, stream);
```

Position embeddings use smaller variance:

```cpp
// Position: [max_position, d_model]  
float xavier_pos_stddev = sqrt(2.0f / (max_position + d_model));
launchXavierInit(position_embeddings, max_position * d_model, xavier_pos_stddev, seed+1, stream);
```

---

## 6. Backward Pass & Gradient Flow

### 6.1 Embedding Backward

For the embedding lookup backward, we scatter-add gradients to the embedding matrix:

$$\frac{\partial L}{\partial \mathbf{W}_E}[t_i, :] \mathrel{+}= \frac{\partial L}{\partial \mathbf{e}_i}$$

**Challenge:** Multiple tokens may map to the same embedding row, requiring atomic operations.

### 6.2 GPU Backward Kernel

```cpp
__global__ void EmbeddingBackwardKernel(const float* grad_output,
                                        const int* token_ids,
                                        float* grad_embeddings,
                                        int total_tokens,
                                        int d_model,
                                        int vocab_size) {
    const int token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= total_tokens) return;

    const int token_id = token_ids[token_index];
    if (token_id < 0 || token_id >= vocab_size) return;

    const float* grad_src = grad_output + token_index * d_model;
    float* grad_dst = grad_embeddings + token_id * d_model;

    for (int i = 0; i < d_model; ++i) {
        atomicAdd(&grad_dst[i], grad_src[i]);
    }
}
```

**Grid Configuration:**
- Block: `(256)` threads
- Grid: `ceil(total_tokens / 256)` blocks
- Each thread processes one token's gradient contribution

### 6.3 RoPE Backward

RoPE rotation is orthogonal, so the backward pass applies the **inverse rotation**:

$$R(-\theta) = R(\theta)^T = R(\theta)^{-1}$$

Implementation negates the sine term:

```cpp
__device__ __forceinline__ void applyInverseRotation(float& x, float& y,
                                                     float cos_val, float sin_val) {
    // Inverse: negate sin to rotate by -theta
    const float x_unrot = x * cos_val + y * sin_val;   // x*cos + y*sin
    const float y_unrot = -x * sin_val + y * cos_val;  // -x*sin + y*cos
    x = x_unrot;
    y = y_unrot;
}
```

**Mathematical Verification:**
$$R(-\theta) \cdot R(\theta) = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = I$$

---

## 7. GPU Implementation Details

### 7.1 Forward Kernel Variants

GRIM-text provides two forward kernels:

| Kernel | RMSNorm | Passes | Shared Memory |
|--------|---------|--------|---------------|
| `EmbeddingLookupKernel` | No | 1 | None |
| `EmbeddingRMSNormKernel` | Yes | 2 | 256 floats |

### 7.2 Fused Embedding + RMSNorm

When `apply_rms_norm = true`, the kernel fuses lookup with normalization:

```cpp
// Pass 1: Lookup + sum of squares
float sq_sum = 0.0f;
for (int i = tid; i < d_model; i += blockDim.x) {
    float value = token_ptr[i] + pos_ptr[i];
    out_ptr[i] = value;
    sq_sum += value * value;
}

// Block reduction
shared_sum[tid] = sq_sum;
__syncthreads();
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
    __syncthreads();
}

// Compute inv_rms
if (tid == 0) {
    inv_rms = rsqrtf(shared_sum[0] / d_model + epsilon);
}
__syncthreads();

// Pass 2: Normalize
for (int i = tid; i < d_model; i += blockDim.x) {
    out_ptr[i] = out_ptr[i] * inv_rms * gamma[i];
}
```

### 7.3 Runtime Interface

The `EmbeddingRuntime` structure provides stateful inference:

```cpp
struct EmbeddingRuntime {
    EmbeddingConfig config;
    EmbeddingWeights weights;

    // GPU buffers (owned by runtime)
    float* token_buffer;        // [vocab_size, d_model]
    float* position_buffer;     // [max_position, d_model]
    float* gamma_buffer;        // [d_model] for RMSNorm

    // Pre-allocated single-token buffers
    int* single_token_id;       // Avoids allocation per call
    int* single_position;

    cudaStream_t stream;
    bool owns_stream;
};
```

**Inference API:**
```cpp
// Batched forward (training, prompt encoding)
bool embeddingRuntimeForward(runtime, token_ids, positions, batch_size, seq_len, output);

// Single-token forward (incremental generation)
bool embeddingRuntimeForwardSingle(runtime, token_id, position, output);
```

### 7.4 Memory Management

CUDA memory is explicitly managed with ownership tracking:

```cpp
void destroyEmbeddingRuntime(EmbeddingRuntime* runtime) {
    if (!runtime) return;
    
    if (runtime->token_buffer) cudaFree(runtime->token_buffer);
    if (runtime->position_buffer) cudaFree(runtime->position_buffer);
    if (runtime->gamma_buffer) cudaFree(runtime->gamma_buffer);
    if (runtime->single_token_id) cudaFree(runtime->single_token_id);
    if (runtime->single_position) cudaFree(runtime->single_position);
    
    if (runtime->owns_stream && runtime->stream) 
        cudaStreamDestroy(runtime->stream);
    
    delete runtime;
}
```

---

## 8. Serialization Format

### 8.1 FlatBuffers Schema

GRIM-text uses FlatBuffers for zero-copy checkpoint serialization:

```flatbuffers
table EmbeddingWeights {
  token_embeddings: [float] (required);    // [vocab_size × d_model]
  positional_encodings: [float];           // [max_seq_len × d_model], optional
  rms_gamma: [float];                      // [d_model], optional
  
  vocab_size: uint32;
  d_model: uint32;
  max_seq_len: uint32;
  use_rms_norm: bool = false;
}
```

### 8.2 File Format

Checkpoint files (`.bin`, `GRMT` identifier):

| Section | Contents |
|---------|----------|
| Header | FlatBuffer metadata, version, checksums |
| ModelConfig | Architecture hyperparameters |
| EmbeddingWeights | Token/position embeddings |
| EncoderLayers | Per-layer attention + FFN weights |
| LMHeadWeights | Output projection (or reference to embeddings if tied) |
| TrainingMetadata | Optimizer state, loss history |

### 8.3 Weight Tying in Serialization

When `tie_embeddings = true`:

```flatbuffers
table LMHeadWeights {
  projection_data: [float];  // NULL when tied
  d_model: uint32;
  vocab_size: uint32;
  tie_embeddings: bool;      // true
}
```

The loader detects `projection_data == null && tie_embeddings == true` and aliases pointers:

```cpp
if (lm_head->tie_embeddings()) {
    model.lm_head_weights = model.embedding_weights.token_embeddings;
    model.lm_head_weights_owned = false;
}
```

---

## 9. API Reference

### 9.1 Forward Pass

```cpp
// Kernel launcher
void launchEmbeddingLookup(const EmbeddingForwardArgs& args,
                           const EmbeddingConfig& config);

// Arguments
struct EmbeddingForwardArgs {
    const int* token_ids;     // [batch × seq_len]
    const int* positions;     // [batch × seq_len] or nullptr
    int batch_size;
    int seq_len;
    float* output;            // [batch × seq_len, d_model]
    const EmbeddingWeights* weights;
    cudaStream_t stream;
};
```

**Error Handling (Rule 20):**
- Throws `std::runtime_error` on null pointers
- Throws on invalid dimensions (≤0)
- Throws on null stream (default stream disallowed)

### 9.2 Backward Pass

```cpp
void launchEmbeddingBackward(const float* grad_output,
                             const int* token_ids,
                             float* grad_embeddings,
                             int batch_size,
                             int seq_len,
                             int d_model,
                             int vocab_size,
                             cudaStream_t stream);
```

**Notes:**
- `grad_embeddings` must be pre-zeroed before accumulation
- Uses `atomicAdd` for thread-safe gradient scatter

### 9.3 Positional Bias Method

```cpp
// Initialize PBM state
bool initializePBM(const PBMConfig& config, PBMState& state);

// Ensure state matches config (re-init if needed)
bool ensurePBM(const PBMConfig& config, PBMState& state);

// Release GPU memory
void releasePBM(PBMState& state);

// Get view for Flash Attention
PBMSpec getPBMSpec(const PBMState& state);

// GQA-aware RoPE rotation (applied to Q, K before attention)
// NOTE: Use this for ALL cases - set num_q_heads == num_kv_heads for MHA
void launchRoPERotationGQA(float* Q, float* K, const float* inv_freq,
                           int batch_size, int num_q_heads, int num_kv_heads,
                           int seq_len, int head_dim, int rotary_dim,
                           cudaStream_t stream);

// RoPE backward (inverse rotation for gradients)
void launchRoPERotationGQA_backward(float* grad_Q, float* grad_K,
                                    const float* inv_freq,
                                    int batch_size, int num_q_heads, int num_kv_heads,
                                    int seq_len, int head_dim, int rotary_dim,
                                    cudaStream_t stream);
```

### 9.4 Initialization

```cpp
// Xavier normal (Gaussian)
void launchXavierNormal(const XavierInitArgs& args);

// Xavier uniform
void launchXavierUniform(const XavierInitArgs& args);

// Convenience overload
void launchXavierInit(float* weights, int size, float stddev,
                      unsigned int seed, cudaStream_t stream);
```

---

## 10. Performance Characteristics

### 10.1 Memory Requirements

| Component | Size | Formula |
|-----------|------|---------|
| Token embeddings | 115 MB | `37555 × 768 × 4 bytes` |
| Position embeddings | 6.3 MB | `2048 × 768 × 4 bytes` |
| ALiBi slopes | 48 B | `12 × 4 bytes` |
| RoPE inv_freq | 128 B | `32 × 4 bytes` |
| **Total** | ~122 MB | |

*Based on default config: vocab=37,555, d_model=768, max_seq=2048, num_heads=12, head_dim=64*

### 10.2 Computational Complexity

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Embedding lookup | O(B × S × d) | Memory-bound |
| Position add | O(B × S × d) | Fused with lookup |
| RMSNorm | O(2 × B × S × d) | Two passes |
| RoPE rotation | O(B × H × S × d/2) | Per head pair |

### 10.3 Kernel Launch Configuration

| Kernel | Block Size | Grid Size |
|--------|------------|-----------|
| EmbeddingLookup | 256 | total_tokens |
| EmbeddingRMSNorm | 256 | total_tokens |
| EmbeddingBackward | 256 | ceil(total_tokens/256) |
| RoPERotation | 256 | (ceil(S/256), H, B) |

### 10.4 Memory Bandwidth

Embedding operations are **memory-bound**:

```
Theoretical bandwidth (RTX 3080): 760 GB/s
Embedding forward: ~450 GB/s achieved
RoPE rotation: ~400 GB/s achieved
```

### 10.5 Optimization Notes

1. **Batched Position Computation**: Auto-position (`token_idx % seq_len`) avoids position array allocation
2. **Fused Kernels**: Embedding + RMSNorm fusion reduces global memory traffic
3. **Pre-allocated Buffers**: `EmbeddingRuntime` avoids per-call allocations
4. **Coalesced Access**: Row-major layout ensures contiguous memory reads

---

## Appendix A: Key Source Files

| File | Purpose |
|------|---------|
| `Layers/Embedding/Embedding_GPU.hpp` | Public API and structures |
| `Layers/Embedding/Embedding_GPU.cu` | CUDA kernels and launchers |
| `Shared/PBM/PositionalBiasMethod.hpp` | PBM API (ALiBi + RoPE) |
| `Shared/PBM/PositionalBiasMethod.cu` | PBM kernel implementations |
| `Shared/Activations/Xavier/Xavier.hpp` | Weight initialization API |
| `Shared/Activations/Xavier/Xavier.cu` | Xavier kernel implementations |
| `Shared/HyperParameters/HyperParameters_GPU.hpp` | System constants |
| `training/schemas/grim_transformer_model.fbs` | FlatBuffer schema |

---

## Appendix B: Configuration Constants

```cpp
// From HyperParameters_GPU.hpp

// Model defaults
constexpr int DEFAULT_D_MODEL = 768;
constexpr int DEFAULT_NUM_HEADS = 12;
constexpr int DEFAULT_NUM_KV_HEADS = 4;     // GQA: 3:1 ratio
constexpr int DEFAULT_MAX_SEQ_LEN = 2048;

// Numerical stability
constexpr float EPSILON_RMSNORM = 1e-5f;

// RoPE defaults
constexpr float ROPE_THETA = 10000.0f;      // Base frequency
constexpr float ROPE_SCALING = 1.0f;        // NTK scaling (1.0 = none)

// ALiBi defaults
constexpr float ALIBI_SLOPE_EXPONENT = -8.0f;

// Token ranges
constexpr int BYTE_TOKEN_END = 256;
constexpr int ATOM_TOKEN_START = 256;
constexpr int NUM_ATOM_TYPES = 19;
```

---

## Appendix C: Mathematical Derivations

### C.1 Xavier Initialization Derivation

Goal: Maintain variance through layers.

For layer $\mathbf{y} = \mathbf{W}\mathbf{x}$:

$$\text{Var}(y_i) = \text{Var}\left(\sum_{j=1}^{n_{in}} W_{ij} x_j\right) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

To maintain $\text{Var}(y) = \text{Var}(x)$:

$$\text{Var}(W) = \frac{1}{n_{in}}$$

Considering both forward ($n_{in}$) and backward ($n_{out}$):

$$\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$$

### C.2 RoPE Inner Product Preservation

For vectors $\mathbf{q}_m, \mathbf{k}_n$ at positions $m, n$:

$$\langle R_{\theta,m}\mathbf{q}_m, R_{\theta,n}\mathbf{k}_n \rangle = \langle R_{\theta,m-n}\mathbf{q}_m, \mathbf{k}_n \rangle$$

The inner product depends only on **relative** position $(m-n)$, not absolute positions.

### C.3 ALiBi Extrapolation Property

ALiBi's linear bias enables extrapolation beyond training lengths:

$$\text{bias}(i, j) = m \cdot |i - j|$$

Since bias grows linearly with distance, the model can handle any sequence length without learned position embeddings breaking down.

---

*Document prepared for academic review. For implementation questions, see the referenced source files.*
