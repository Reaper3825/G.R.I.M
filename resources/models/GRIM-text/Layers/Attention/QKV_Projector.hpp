#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace GRIM {

struct QKVProjectionConfig {
    int d_model = 0;
    int num_heads = 0;
    int num_kv_heads = 0;  // GQA: K,V head count (0 = use num_heads for MHA mode)
    int batch_size = 1;
    int seq_len = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    
    // Helper: Get effective number of KV heads (defaults to num_heads for MHA)
    int getNumKVHeads() const {
        return (num_kv_heads > 0) ? num_kv_heads : num_heads;
    }
    
    // Helper: Compute Q projection size
    int getQProjectionSize() const {
        return d_model;  // num_heads * head_dim = d_model
    }
    
    // Helper: Compute K or V projection size (reduced in GQA)
    int getKVProjectionSize() const {
        int head_dim = (num_heads > 0) ? (d_model / num_heads) : 64;
        return getNumKVHeads() * head_dim;
    }
    
    // Helper: Check if GQA is active
    bool isGQA() const {
        return num_kv_heads > 0 && num_kv_heads < num_heads;
    }
};

struct QKVProjectionWeights {
    // For MHA: W_qkv is [3 * d_model, d_model], b_qkv is [3 * d_model]
    // For GQA: W_q is [d_model, d_model], W_kv is [2 * kv_dim, d_model]
    //         b_q is [d_model], b_kv is [2 * kv_dim]
    // Using unified layout: W_qkv = [d_model + 2*kv_dim, d_model]
    const float* W_qkv = nullptr; // [q_size + 2*kv_size, d_model] row-major
    const float* b_qkv = nullptr; // [q_size + 2*kv_size]
    
    // Alternative: Separate weights for GQA (optional, for cleaner code)
    const float* W_q = nullptr;   // [d_model, d_model] - Q projection
    const float* W_k = nullptr;   // [kv_dim, d_model] - K projection
    const float* W_v = nullptr;   // [kv_dim, d_model] - V projection
    const float* b_q = nullptr;   // [d_model]
    const float* b_k = nullptr;   // [kv_dim]
    const float* b_v = nullptr;   // [kv_dim]
};

size_t getQkvProjectionWorkspaceSize(const QKVProjectionConfig& config);

// Standard projection: uses W_qkv unified format
void launchQkvProjection(const float* input,
                         const QKVProjectionWeights& weights,
                         float* q_out,
                         float* k_out,
                         float* v_out,
                         float* workspace,
                         const QKVProjectionConfig& config);

// GQA projection: uses separate W_q, W_k, W_v weights
// This is more memory-efficient for GQA as it avoids fused projection
void launchGQAProjection(const float* input,
                         const QKVProjectionWeights& weights,
                         float* q_out,
                         float* k_out,
                         float* v_out,
                         const QKVProjectionConfig& config);

// Reshape Q, K, V from [tokens, dim] to [batch, heads, seq, head_dim]
// Used to convert QKV projection output to Flash Attention input format
// For GQA: Q uses num_heads, K/V use num_kv_heads
void launchQKVReshapeToBHSD(const float* q_in,
                            const float* k_in,
                            const float* v_in,
                            float* q_out,
                            float* k_out,
                            float* v_out,
                            int batch_size,
                            int seq_len,
                            int num_heads,
                            int num_kv_heads,  // GQA: K,V head count
                            int head_dim,
                            cudaStream_t stream);



// Reshape single tensor from [tokens, d_model] to [batch, heads, seq, head_dim]
// Used to convert grad_concat to BHSD format for attention backward pass
void launchReshapeToBHSD(const float* src,
                         float* dst,
                         int batch_size,
                         int seq_len,
                         int num_heads,
                         int head_dim,
                         cudaStream_t stream);

// Reshape from [batch, heads, seq, head_dim] back to [tokens, d_model]
// Used to convert Flash Attention output to standard format
void launchReshapeFromBHSD(const float* src,
                           float* dst,
                           int batch_size,
                           int seq_len,
                           int num_heads,
                           int head_dim,
                           cudaStream_t stream);

} // namespace GRIM
