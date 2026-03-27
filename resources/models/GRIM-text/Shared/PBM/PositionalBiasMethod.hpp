//======================================================//
//  Shared/PBM/PositionalBiasMethod.hpp
//  Unified Positional Bias Method: ALiBi + RoPE Hybrid
//  
//  This is the ONLY PBM implementation. Always applies both:
//  - ALiBi: Attention-Linear-Biases (slopes added to QK scores)
//  - RoPE: Rotary Position Embeddings (rotation applied to Q,K)
//
//  Design: No mode switching. Hybrid is always enabled.
//  If you only want one, use the PyTorch reference implementation.
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "../HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM::PBM {

// ═══════════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════════

struct PBMConfig {
    // ALiBi configuration (defaults from HyperParameters)
    int num_heads = GRIM::HyperParameters::DEFAULT_NUM_HEADS;
    float alibi_slope_exponent = GRIM::HyperParameters::ALIBI_SLOPE_EXPONENT;
    
    // ISSUE #78 FIX: Maximum ALiBi bias cap (negative value, e.g., -10.0f)
    // Slopes are capped so that: abs(slope) * max_seq_len <= abs(alibi_max_bias)
    // Set to 0.0f to disable capping (NOT RECOMMENDED)
    float alibi_max_bias = GRIM::HyperParameters::ALIBI_MAX_BIAS;
    
    // Context length for ALiBi slope calculation (must be set from model config)
    // This determines the maximum position distance ALiBi is calibrated for
    int max_seq_len = GRIM::HyperParameters::DEFAULT_MAX_SEQ_LEN;
    
    // RoPE configuration
    // head_dim: MUST be set from LanguageModelConfig.head_dim (= d_model / num_heads)
    // DO NOT use DEFAULT_HEAD_DIM - it may not match actual model dimensions
    int head_dim = 0;  // REQUIRED - set from config.head_dim
    int rotary_dim = 0; // Usually same as head_dim, set from config.head_dim
    float rope_theta = GRIM::HyperParameters::ROPE_THETA;
    float rope_scaling = GRIM::HyperParameters::ROPE_SCALING;
    
    // GQA support (defaults from HyperParameters)
    int num_kv_heads = GRIM::HyperParameters::DEFAULT_NUM_KV_HEADS;
    
    // Runtime
    cudaStream_t stream = nullptr;
    bool verbose = false;
};

// ═══════════════════════════════════════════════════════════════════════════
//  State (GPU buffers)
// ═══════════════════════════════════════════════════════════════════════════

struct PBMState {
    // ALiBi state
    float* alibi_slopes = nullptr;       // Device: [num_heads] slopes
    std::vector<float> alibi_slopes_host;
    int num_heads = 0;
    
    // RoPE state
    float* rope_inv_freq = nullptr;      // Device: [rotary_dim/2] inverse frequencies
    std::vector<float> rope_inv_freq_host;
    int head_dim = 0;
    int rotary_dim = 0;
    
    // GQA
    int num_kv_heads = 0;
    
    // Config values used during initialization (for stale reuse detection)
    int max_seq_len = 0;
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
    float alibi_slope_exponent = 0.0f;
    
    // Status
    bool initialized = false;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Positional Encoding Spec (passed to Flash Attention)
// ═══════════════════════════════════════════════════════════════════════════

struct PBMSpec {
    // RoPE view (for Q,K rotation before attention)
    const float* rope_inv_freq = nullptr;
    int rotary_dim = 0;
    
    // ALiBi view (for attention bias during score computation)
    const float* alibi_slopes = nullptr;
    int num_heads = 0;
    
    // GQA info
    int num_kv_heads = 0;
    
    // Valid flag
    bool valid = false;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Core API
// ═══════════════════════════════════════════════════════════════════════════

// Initialize PBM state (allocates GPU buffers, computes slopes/frequencies)
bool initializePBM(const PBMConfig& config, PBMState& state);

// Ensure state matches config (re-initializes if dimensions changed)
bool ensurePBM(const PBMConfig& config, PBMState& state);

// Release GPU memory
void releasePBM(PBMState& state);

// Get spec for Flash Attention (views into state)
PBMSpec getPBMSpec(const PBMState& state);

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE Rotation Kernels (applied to Q,K before attention)
//  NOTE: Only GQA versions are provided. For MHA (num_heads == num_kv_heads),
//  call the GQA version with num_q_heads == num_kv_heads.
// ═══════════════════════════════════════════════════════════════════════════

// GQA-aware RoPE rotation (handles both GQA and MHA cases)
// Q: [batch, num_q_heads, seq_len, head_dim]
// K: [batch, num_kv_heads, seq_len, head_dim]
// For MHA: set num_q_heads == num_kv_heads
void launchRoPERotationGQA(
    float* Q,                           // Query tensor (in-place)
    float* K,                           // Key tensor (in-place)
    const float* inv_freq,              // Inverse frequencies [rotary_dim/2]
    int batch_size,
    int num_q_heads,                    // Q head count (larger in GQA)
    int num_kv_heads,                   // KV head count (smaller in GQA)
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream = nullptr,
    int pos_offset = 0                  // Position offset for KV cache decode (default 0)
);

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE Backward Kernels (inverse rotation for gradient propagation)
//  
//  CRITICAL: RoPE forward rotates Q,K by angle θ. During backward, gradients
//  from Flash Attention are in the ROTATED coordinate space. We must apply
//  the INVERSE rotation (-θ) to transform gradients back to original space.
//  
//  Math: Forward uses R(θ), Backward uses R(-θ) = R(θ)^T
//  Implementation: Same formula but with sin_theta negated
// ═══════════════════════════════════════════════════════════════════════════

// GQA-aware RoPE backward (inverse rotation for gradients)
// grad_Q: [batch, num_q_heads, seq_len, head_dim] - gradients in rotated space
// grad_K: [batch, num_kv_heads, seq_len, head_dim] - gradients in rotated space
// After call: grad_Q, grad_K are in original (unrotated) space
void launchRoPERotationGQA_backward(
    float* grad_Q,                      // Query gradient tensor (in-place)
    float* grad_K,                      // Key gradient tensor (in-place)
    const float* inv_freq,              // Inverse frequencies [rotary_dim/2]
    int batch_size,
    int num_q_heads,                    // Q head count (larger in GQA)
    int num_kv_heads,                   // KV head count (smaller in GQA)
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream = nullptr
);

// ═══════════════════════════════════════════════════════════════════════════
//  Memory Helpers
// ═══════════════════════════════════════════════════════════════════════════

// Returns GPU bytes required for PBM buffers
inline size_t getPBMDeviceBytes(const PBMConfig& config) {
    const size_t alibi_bytes = static_cast<size_t>(config.num_heads) * sizeof(float);
    const size_t rope_bytes = static_cast<size_t>(config.rotary_dim / 2) * sizeof(float);
    return alibi_bytes + rope_bytes;
}

// Convenience accessors
inline const float* getAlibiSlopes(const PBMState& state) {
    return state.initialized ? state.alibi_slopes : nullptr;
}

inline const float* getRoPEInvFreq(const PBMState& state) {
    return state.initialized ? state.rope_inv_freq : nullptr;
}

inline int getRotaryDimension(const PBMState& state) {
    return state.initialized ? state.rotary_dim : 0;
}

} // namespace GRIM::PBM
