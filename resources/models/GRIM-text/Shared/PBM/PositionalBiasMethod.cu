//======================================================//
//  Shared/PBM/PositionalBiasMethod.cu
//  Unified Positional Bias Method Implementation
//  
//  Combines ALiBi slope computation and RoPE inverse frequencies
//  into a single initialization path. Both are ALWAYS enabled.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include "PositionalBiasMethod.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace GRIM::PBM {

namespace {

constexpr const char* kTag = "[PBM]";

bool checkCuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::cerr << kTag << " " << what << " failed: " << cudaGetErrorString(err) << std::endl;
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ALiBi Slope Computation (Host) - Context-Aware Scaling
//  
//  This formula explicitly controls penalty strength at near vs far distances
//  and automatically adapts when context length changes.
//  
//  Parameters:
//    d_min = locality distance (tied to rotary_dim/2 or minimum 16)
//    d_max = context length (max_seq_len)
//    target_bias = |alibi_slope_exponent| (interpreted as MAX bias at max_seq_len)
//  
//  Formula (scaled to enforce max bias at max_seq_len for strongest head):
//    base_m_max = target_bias / d_min
//    base_m_min = target_bias / d_max
//    scale = d_min / d_max
//    m_max = base_m_max * scale  (strongest head reaches -target_bias at max distance)
//    m_min = base_m_min * scale  (weakest head is proportionally gentler)
//    slopes interpolated geometrically across heads
//  
//  WARNING: If you scale slopes for max_seq_len=2048 but run 8192 tokens,
//           your weakest heads will be too weak. Always set max_seq_len correctly!
// ═══════════════════════════════════════════════════════════════════════════

bool computeAlibiSlopes(const PBMConfig& config, std::vector<float>& out_slopes) {
    if (config.num_heads <= 0) {
        std::cerr << kTag << " Invalid num_heads=" << config.num_heads << std::endl;
        return false;
    }
    
    if (config.max_seq_len <= 0) {
        std::cerr << kTag << " Invalid max_seq_len=" << config.max_seq_len 
                  << " - MUST be set from model config for proper ALiBi scaling!" << std::endl;
        return false;
    }
    
    // d_max = context length (what we're calibrating ALiBi for)
    const int d_max = config.max_seq_len;
    
    // d_min = locality distance (tie to rotary_dim for hybrid ALiBi+RoPE, else use 16)
    // This controls how aggressively the STRONGEST head penalizes distance
    int d_min = config.rotary_dim > 0 ? (config.rotary_dim / 2) : 16;
    d_min = std::max(16, std::min(d_min, d_max));  // Clamp to [16, d_max]
    
    // target_bias controls overall penalty strength (standard ALiBi uses 8.0)
    // Derived from the canonical alibi_slope_exponent value
    const float target_bias = std::abs(config.alibi_slope_exponent);
    
    // Safety check: target_bias==0 would generate all-zero slopes (useless)
    if (target_bias == 0.0f) {
        std::cerr << kTag << " ERROR: alibi_slope_exponent=0 generates all-zero slopes!" << std::endl;
        return false;
    }
    
    // Compute slope range based on context length
    // m_max: strongest head (reaches -target_bias at max distance)
    // m_min: weakest head (only penalizes very distant tokens)
    const float base_m_max = target_bias / static_cast<float>(d_min);
    const float base_m_min = target_bias / static_cast<float>(d_max);
    const float max_bias_scale = static_cast<float>(d_min) / static_cast<float>(d_max);
    const float m_max = base_m_max * max_bias_scale;
    const float m_min = base_m_min * max_bias_scale;
    
    // Safety check: m_min > m_max means d_min > d_max (inverted, should never happen)
    if (m_min > m_max) {
        std::cerr << kTag << " ERROR: m_min(" << m_min << ") > m_max(" << m_max 
                  << ") - d_min/d_max inverted! d_min=" << d_min << ", d_max=" << d_max << std::endl;
        return false;
    }
    
    // ISSUE #76: Log context-aware ALiBi penalty computation for debugging
    // By design, head 0 reaches -target_bias at max_seq_len (not -target_bias * d_max/d_min).
    // Example: max_seq_len=1024, d_min=32, target_bias=8
    //   base_m_max = 8/32 = 0.25, scale = 32/1024 = 0.03125
    //   m_max = 0.25 * 0.03125 = 0.0078125
    //   At position 1024, bias = -0.0078125 * 1024 = -8.0 (as designed)
    std::cerr << "[PBM-ALIBI-ISSUE76] max_seq_len=" << config.max_seq_len 
              << " d_min=" << d_min 
              << " target_bias=" << target_bias
              << " base_m_max=" << base_m_max
              << " base_m_min=" << base_m_min
              << " scale=" << max_bias_scale
              << " => m_max=" << m_max << " m_min=" << m_min << std::endl;
    std::cerr << "[PBM-ALIBI-ISSUE76] At position " << config.max_seq_len << ":\n"
              << "    Head 0 (strongest): bias = -" << m_max << " * " << config.max_seq_len 
              << " = " << (-m_max * config.max_seq_len) << std::endl;
    std::cerr << "    Head " << (config.num_heads - 1) << " (weakest): bias = -" << m_min << " * " << config.max_seq_len 
              << " = " << (-m_min * config.max_seq_len) << std::endl;
    
    out_slopes.resize(static_cast<size_t>(config.num_heads));
    
    // ISSUE #78 FIX: Compute maximum allowed slope magnitude based on bias cap
    // If alibi_max_bias = -10.0 and max_seq_len = 1024:
    //   max_slope_magnitude = |-10| / 1024 = 0.00976
    // Any slope with larger magnitude will cause bias to exceed -10 at max distance
    const float max_bias_magnitude = std::abs(config.alibi_max_bias);
    const float max_slope_magnitude = (config.alibi_max_bias != 0.0f && config.max_seq_len > 0)
        ? (max_bias_magnitude / static_cast<float>(config.max_seq_len))
        : 0.0f;  // 0 = no capping
    
    if (config.num_heads == 1) {
        // Single head: use strongest slope (with optional capping) 
        float slope = -m_max;
        if (max_slope_magnitude > 0.0f && m_max > max_slope_magnitude) {
            slope = -max_slope_magnitude;
            std::cerr << "[PBM-ALIBI-ISSUE78] Head 0 slope CAPPED: -" << m_max 
                      << " -> " << slope << " (max_bias=" << config.alibi_max_bias << ")" << std::endl;
        }
        out_slopes[0] = slope;
        return true;
    }
    
    // Interpolate slopes geometrically from m_max to m_min
    // Head 0 = strongest (m_max), Head N-1 = weakest (m_min)
    const float log_mmax = std::log(m_max);
    const float log_mmin = std::log(m_min);
    
    int capped_count = 0;
    for (int h = 0; h < config.num_heads; ++h) {
        const float t = static_cast<float>(h) / static_cast<float>(config.num_heads - 1);
        const float log_m = log_mmax + t * (log_mmin - log_mmax);
        float m = std::exp(log_m);
        
        // ISSUE #78 FIX: Cap slope magnitude to prevent extreme biases
        // slope * max_seq_len should not exceed alibi_max_bias
        if (max_slope_magnitude > 0.0f && m > max_slope_magnitude) {
            m = max_slope_magnitude;
            ++capped_count;
        }
        
        // ISSUE #69 FIX: FlashAttention library expects NEGATIVE slopes
        // (it uses += slope * col_idx for causal attention)
        out_slopes[static_cast<size_t>(h)] = -m;
    }
    
    // Issue #78: Log how many heads were capped
    if (capped_count > 0) {
        std::cerr << "[PBM-ALIBI-ISSUE78] " << capped_count << "/" << config.num_heads 
                  << " heads CAPPED to max_slope=" << max_slope_magnitude 
                  << " (max_bias=" << config.alibi_max_bias 
                  << " at max_seq_len=" << config.max_seq_len << ")" << std::endl;
        std::cerr << "[PBM-ALIBI-ISSUE78] Effective bias range: [" << config.alibi_max_bias 
                  << ", 0] instead of [" << (-m_max * config.max_seq_len) << ", 0]" << std::endl;
    }
    
    if (config.verbose) {
        std::cout << kTag << " ALiBi slopes computed (context-aware scaling):" << std::endl;
        std::cout << "    max_seq_len=" << config.max_seq_len 
                  << ", d_min=" << d_min 
                  << ", target_bias=" << target_bias << std::endl;
        std::cout << "    m_max=" << m_max << " (head 0), m_min=" << m_min 
                  << " (head " << (config.num_heads-1) << ")" << std::endl;
        if (capped_count > 0) {
            std::cout << "    ISSUE #78: " << capped_count << " heads capped to max_slope=" 
                      << max_slope_magnitude << std::endl;
        }
        const int preview = std::min(config.num_heads, 4);
        for (int h = 0; h < preview; ++h) {
            std::cout << "    head[" << h << "] slope=" << out_slopes[static_cast<size_t>(h)] << std::endl;
        }
        if (config.num_heads > 4) {
            std::cout << "    ... (" << (config.num_heads - 4) << " more heads)" << std::endl;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE Inverse Frequency Computation (Host) - Context-Aware NTK Scaling
//  
//  Standard Formula: inv_freq[i] = 1.0 / theta^(2*i / rotary_dim)
//  
//  NTK-Aware Scaling (for extending context beyond training length):
//    When max_seq_len > base_seq_len (default 2048):
//    - scaled_theta = theta * (max_seq_len / base_seq_len)^(rotary_dim / (rotary_dim - 2))
//    - This adjusts the rotation frequencies to accommodate longer sequences
//    - Equivalent to "dynamic NTK" / "Code Llama" style scaling
//  
//  rope_scaling field provides additional manual scaling (1.0 = no extra scaling)
//  
//  WARNING: If you set max_seq_len=2048 but run inference at 8192 tokens,
//           position encodings will be extrapolated (may degrade quality)!
// ═══════════════════════════════════════════════════════════════════════════

bool computeRoPEInvFreq(const PBMConfig& config, std::vector<float>& out_inv_freq) {
    if (config.rotary_dim <= 0 || (config.rotary_dim & 1) != 0) {
        std::cerr << kTag << " Invalid rotary_dim=" << config.rotary_dim 
                  << " (must be positive and even)" << std::endl;
        return false;
    }
    
    if (config.max_seq_len <= 0) {
        std::cerr << kTag << " Invalid max_seq_len=" << config.max_seq_len 
                  << " - MUST be set from model config for proper RoPE scaling!" << std::endl;
        return false;
    }
    
    const int half_dim = config.rotary_dim / 2;
    out_inv_freq.resize(static_cast<size_t>(half_dim));
    
    // Base context length for which theta was originally calibrated (standard: 2048)
    constexpr int BASE_SEQ_LEN = 2048;
    
    // Compute NTK-aware theta scaling if extending beyond base context
    float effective_theta = config.rope_theta;
    if (config.max_seq_len > BASE_SEQ_LEN && config.rotary_dim > 2) {
        // NTK-aware formula: theta_scaled = theta * (ctx / base)^(dim / (dim - 2))
        const float ctx_ratio = static_cast<float>(config.max_seq_len) / static_cast<float>(BASE_SEQ_LEN);
        const float ntk_exponent = static_cast<float>(config.rotary_dim) / 
                                   static_cast<float>(config.rotary_dim - 2);
        effective_theta = config.rope_theta * std::pow(ctx_ratio, ntk_exponent);
    }
    
    // Safety check: effective_theta must be positive for valid rotation frequencies
    if (effective_theta <= 0.0f) {
        std::cerr << kTag << " ERROR: effective_theta=" << effective_theta 
                  << " (must be > 0). Check rope_theta=" << config.rope_theta << std::endl;
        return false;
    }
    
    for (int i = 0; i < half_dim; ++i) {
        const float exp_arg = static_cast<float>(2 * i) / static_cast<float>(config.rotary_dim);
        // Apply rope_scaling for any additional manual scaling (typically 1.0)
        out_inv_freq[static_cast<size_t>(i)] = config.rope_scaling / 
            std::pow(effective_theta, exp_arg);
    }
    
    if (config.verbose) {
        std::cout << kTag << " RoPE inverse frequencies computed (context-aware NTK scaling):" << std::endl;
        std::cout << "    rotary_dim=" << config.rotary_dim 
                  << ", max_seq_len=" << config.max_seq_len << std::endl;
        std::cout << "    base_theta=" << config.rope_theta
                  << ", effective_theta=" << effective_theta
                  << ", manual_scaling=" << config.rope_scaling << std::endl;
        if (config.max_seq_len > BASE_SEQ_LEN) {
            std::cout << "    NTK scaling active: extending from " << BASE_SEQ_LEN 
                      << " to " << config.max_seq_len << " context" << std::endl;
        }
        const int preview = std::min(half_dim, 4);
        for (int i = 0; i < preview; ++i) {
            std::cout << "    inv_freq[" << i << "]=" << out_inv_freq[static_cast<size_t>(i)] << std::endl;
        }
        if (half_dim > 4) {
            std::cout << "    ... (" << (half_dim - 4) << " more frequencies)" << std::endl;
        }
    }
    return true;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Core API Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool initializePBM(const PBMConfig& config, PBMState& state) {
    // Clean up any existing state
    releasePBM(state);
    
    std::cout << kTag << " Initializing Hybrid ALiBi+RoPE..." << std::endl;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Step 1: Compute ALiBi slopes on host
    // ─────────────────────────────────────────────────────────────────────────
    if (!computeAlibiSlopes(config, state.alibi_slopes_host)) {
        std::cerr << kTag << " Failed to compute ALiBi slopes" << std::endl;
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Step 2: Compute RoPE inverse frequencies on host
    // ─────────────────────────────────────────────────────────────────────────
    if (!computeRoPEInvFreq(config, state.rope_inv_freq_host)) {
        std::cerr << kTag << " Failed to compute RoPE inverse frequencies" << std::endl;
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Step 3: Allocate GPU buffers
    // ─────────────────────────────────────────────────────────────────────────
    const size_t alibi_bytes = state.alibi_slopes_host.size() * sizeof(float);
    const size_t rope_bytes = state.rope_inv_freq_host.size() * sizeof(float);
    
    if (!checkCuda(cudaMalloc(&state.alibi_slopes, alibi_bytes), "cudaMalloc(alibi_slopes)")) {
        return false;
    }
    
    if (!checkCuda(cudaMalloc(&state.rope_inv_freq, rope_bytes), "cudaMalloc(rope_inv_freq)")) {
        cudaFree(state.alibi_slopes);
        state.alibi_slopes = nullptr;
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Step 4: Upload to GPU
    // ─────────────────────────────────────────────────────────────────────────
    if (config.stream) {
        // Async upload
        if (!checkCuda(cudaMemcpyAsync(state.alibi_slopes, state.alibi_slopes_host.data(),
                                        alibi_bytes, cudaMemcpyHostToDevice, config.stream),
                       "cudaMemcpyAsync(alibi_slopes)")) {
            releasePBM(state);
            return false;
        }
        if (!checkCuda(cudaMemcpyAsync(state.rope_inv_freq, state.rope_inv_freq_host.data(),
                                        rope_bytes, cudaMemcpyHostToDevice, config.stream),
                       "cudaMemcpyAsync(rope_inv_freq)")) {
            releasePBM(state);
            return false;
        }
    } else {
        // Sync upload
        if (!checkCuda(cudaMemcpy(state.alibi_slopes, state.alibi_slopes_host.data(),
                                   alibi_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(alibi_slopes)")) {
            releasePBM(state);
            return false;
        }
        if (!checkCuda(cudaMemcpy(state.rope_inv_freq, state.rope_inv_freq_host.data(),
                                   rope_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(rope_inv_freq)")) {
            releasePBM(state);
            return false;
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Step 5: Store metadata (including config values for stale reuse detection)
    // ─────────────────────────────────────────────────────────────────────────
    state.num_heads = config.num_heads;
    state.head_dim = config.head_dim;
    state.rotary_dim = config.rotary_dim;
    state.num_kv_heads = config.num_kv_heads;
    state.max_seq_len = config.max_seq_len;
    state.rope_theta = config.rope_theta;
    state.rope_scaling = config.rope_scaling;
    state.alibi_slope_exponent = config.alibi_slope_exponent;
    state.alibi_max_bias = config.alibi_max_bias;
    
    // Record event after async upload for cross-stream safety
    if (config.stream) {
        if (state.upload_event == nullptr) {
            if (!checkCuda(cudaEventCreateWithFlags(&state.upload_event, cudaEventDisableTiming),
                           "cudaEventCreate(upload_event)")) {
                releasePBM(state);
                return false;
            }
        }
        if (!checkCuda(cudaEventRecord(state.upload_event, config.stream),
                       "cudaEventRecord(upload_event)")) {
            releasePBM(state);
            return false;
        }
    }
    
    state.initialized = true;
    
    std::cout << kTag << " ✓ Hybrid ALiBi+RoPE initialized successfully" << std::endl;
    std::cout << "    ALiBi: " << state.num_heads << " heads, slopes @ " 
              << (void*)state.alibi_slopes << std::endl;
    std::cout << "    RoPE:  rotary_dim=" << state.rotary_dim 
              << ", inv_freq @ " << (void*)state.rope_inv_freq << std::endl;
    
    return true;
}

bool ensurePBM(const PBMConfig& config, PBMState& state) {
    // Check if re-initialization needed (all config values that affect computation)
    const bool config_match = state.initialized &&
                              state.num_heads == config.num_heads &&
                              state.head_dim == config.head_dim &&
                              state.rotary_dim == config.rotary_dim &&
                              state.num_kv_heads == config.num_kv_heads &&
                              state.max_seq_len == config.max_seq_len &&
                              state.rope_theta == config.rope_theta &&
                              state.rope_scaling == config.rope_scaling &&
                              state.alibi_slope_exponent == config.alibi_slope_exponent &&
                              state.alibi_max_bias == config.alibi_max_bias;
    
    if (config_match) {
        return true;  // Already initialized with matching config
    }
    
    return initializePBM(config, state);
}

void releasePBM(PBMState& state) {
    if (state.alibi_slopes) {
        cudaFree(state.alibi_slopes);
        state.alibi_slopes = nullptr;
    }
    if (state.rope_inv_freq) {
        cudaFree(state.rope_inv_freq);
        state.rope_inv_freq = nullptr;
    }
    if (state.upload_event) {
        cudaEventDestroy(state.upload_event);
        state.upload_event = nullptr;
    }
    
    state.alibi_slopes_host.clear();
    state.alibi_slopes_host.shrink_to_fit();
    state.rope_inv_freq_host.clear();
    state.rope_inv_freq_host.shrink_to_fit();
    
    state.num_heads = 0;
    state.head_dim = 0;
    state.rotary_dim = 0;
    state.num_kv_heads = 0;
    state.max_seq_len = 0;
    state.rope_theta = 0.0f;
    state.rope_scaling = 0.0f;
    state.alibi_slope_exponent = 0.0f;
    state.alibi_max_bias = 0.0f;
    state.initialized = false;
}

PBMSpec getPBMSpec(const PBMState& state) {
    PBMSpec spec{};
    
    if (!state.initialized) {
        spec.valid = false;
        return spec;
    }
    
    spec.rope_inv_freq = state.rope_inv_freq;
    spec.rotary_dim = state.rotary_dim;
    spec.alibi_slopes = state.alibi_slopes;
    spec.num_heads = state.num_heads;
    spec.num_kv_heads = state.num_kv_heads;
    spec.valid = true;
    
    return spec;
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE Rotation Kernels
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// Helper: Apply RoPE rotation to a single pair of dimensions
__device__ __forceinline__ void applyRotation(
    float& x,
    float& y,
    float cos_val,
    float sin_val
) {
    const float x_rot = x * cos_val - y * sin_val;
    const float y_rot = x * sin_val + y * cos_val;
    x = x_rot;
    y = y_rot;
}

// NOTE: ropeRotationKernel (non-GQA) was REMOVED - it was broken for GQA.
// Use ropeRotationGQAKernel for ALL cases (set num_q_heads == num_kv_heads for MHA).

// GQA-aware RoPE kernel: Separate Q and K with different head counts
__global__ void ropeRotationGQAKernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ inv_freq,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    bool is_q_pass,
    int pos_offset
) {
    const int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (pos_idx >= seq_len || batch_idx >= batch_size) return;
    
    const int num_heads = is_q_pass ? num_q_heads : num_kv_heads;
    if (head_idx >= num_heads) return;
    
    const int bhsd_offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    float* tensor = is_q_pass ? Q : K;
    
    const int num_pairs = rotary_dim / 2;
    for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
        const int dim_i = pair_idx * 2;
        const int dim_j = pair_idx * 2 + 1;
        
        const float freq = inv_freq[pair_idx];
        const float theta = static_cast<float>(pos_idx + pos_offset) * freq;
        const float cos_val = cosf(theta);
        const float sin_val = sinf(theta);
        
        float x_i = tensor[bhsd_offset + dim_i];
        float x_j = tensor[bhsd_offset + dim_j];
        applyRotation(x_i, x_j, cos_val, sin_val);
        tensor[bhsd_offset + dim_i] = x_i;
        tensor[bhsd_offset + dim_j] = x_j;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE BACKWARD Kernel (inverse rotation for gradient propagation)
//  
//  Forward: x' = x*cos(θ) - y*sin(θ),  y' = x*sin(θ) + y*cos(θ)
//  Backward (inverse): x = x'*cos(θ) + y'*sin(θ),  y = -x'*sin(θ) + y'*cos(θ)
//  
//  This is equivalent to using sin_val *= -1 in the forward formula.
//  The rotation matrix R(θ) is orthogonal, so R(-θ) = R(θ)^T = R(θ)^(-1)
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ void applyInverseRotation(
    float& x,
    float& y,
    float cos_val,
    float sin_val  // NOTE: This gets negated inside the function
) {
    // Inverse rotation: negate sin to rotate by -theta
    const float neg_sin_val = -sin_val;
    const float x_unrot = x * cos_val - y * neg_sin_val;  // = x*cos + y*sin
    const float y_unrot = x * neg_sin_val + y * cos_val;  // = -x*sin + y*cos
    x = x_unrot;
    y = y_unrot;
}

// GQA-aware RoPE BACKWARD kernel: Inverse rotation for gradient tensors
__global__ void ropeRotationGQABackwardKernel(
    float* __restrict__ grad_Q,
    float* __restrict__ grad_K,
    const float* __restrict__ inv_freq,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    bool is_q_pass,
    int pos_offset
) {
    const int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (pos_idx >= seq_len || batch_idx >= batch_size) return;
    
    const int num_heads = is_q_pass ? num_q_heads : num_kv_heads;
    if (head_idx >= num_heads) return;
    
    const int bhsd_offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    float* tensor = is_q_pass ? grad_Q : grad_K;
    
    const int num_pairs = rotary_dim / 2;
    for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
        const int dim_i = pair_idx * 2;
        const int dim_j = pair_idx * 2 + 1;
        
        const float freq = inv_freq[pair_idx];
        const float theta = static_cast<float>(pos_idx + pos_offset) * freq;
        const float cos_val = cosf(theta);
        const float sin_val = sinf(theta);
        
        float x_i = tensor[bhsd_offset + dim_i];
        float x_j = tensor[bhsd_offset + dim_j];
        applyInverseRotation(x_i, x_j, cos_val, sin_val);
        tensor[bhsd_offset + dim_i] = x_i;
        tensor[bhsd_offset + dim_j] = x_j;
    }
}

} // namespace

// NOTE: Non-GQA launchRoPERotation() was REMOVED (Rule 20: current GQA path only).
// It was broken for GQA (assumed Q and K have same head count, causing memory corruption).
// Use launchRoPERotationGQA() for ALL cases - set num_q_heads == num_kv_heads.

void launchRoPERotationGQA(
    float* Q,
    float* K,
    const float* inv_freq,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream,
    int pos_offset
) {
    // Rule 20: Crash loud on invalid inputs — silent return hides bugs
    if (Q == nullptr || K == nullptr || inv_freq == nullptr) {
        throw std::runtime_error(std::string(kTag) + " Null pointer passed to launchRoPERotationGQA"
            " (Q=" + std::to_string((uintptr_t)Q) +
            " K=" + std::to_string((uintptr_t)K) +
            " inv_freq=" + std::to_string((uintptr_t)inv_freq) + ")");
    }
    
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        throw std::runtime_error(std::string(kTag) + " Invalid rotary_dim=" + std::to_string(rotary_dim)
            + " (head_dim=" + std::to_string(head_dim) + ")");
    }
    
    // Validate GQA configuration: num_q_heads must be divisible by num_kv_heads
    if (num_q_heads <= 0 || num_kv_heads <= 0 || (num_q_heads % num_kv_heads) != 0) {
        throw std::runtime_error(std::string(kTag) + " Invalid GQA config - num_q_heads=" + std::to_string(num_q_heads)
            + " must be divisible by num_kv_heads=" + std::to_string(num_kv_heads));
    }
    
    const int threads_per_block = 256;
    const int blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    // Drain any stale CUDA error from prior unchecked operations (split_qkv,
    // BSM_to_BHSD conversions, cudaFree from temp tensor RAII destructors).
    // Without this drain, cudaGetLastError() after the kernel launch reports
    // the stale error, falsely blaming the RoPE kernel.
    {
        cudaError_t stale = cudaGetLastError();
        if (stale != cudaSuccess) {
            std::cerr << kTag << " WARNING: drained stale CUDA error before RoPE launch: "
                      << cudaGetErrorString(stale)
                      << " (code=" << static_cast<int>(stale) << ")" << std::endl;
        }
    }
    
    // Launch for Q (with num_q_heads)
    {
        dim3 grid(blocks_seq, num_q_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQAKernel<<<grid, block, 0, stream>>>(
            Q, K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            true,  // Q pass
            pos_offset
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << kTag << " Q rotation kernel launch error: " << cudaGetErrorString(err)
                      << " (grid=(" << blocks_seq << "," << num_q_heads << "," << batch_size
                      << ") block=" << threads_per_block
                      << " seq=" << seq_len << " head_dim=" << head_dim
                      << " rotary=" << rotary_dim << ")" << std::endl;
            return;
        }
    }
    
    // Launch for K (with num_kv_heads)
    {
        dim3 grid(blocks_seq, num_kv_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQAKernel<<<grid, block, 0, stream>>>(
            Q, K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            false,  // K pass
            pos_offset
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << kTag << " K rotation kernel launch error: " << cudaGetErrorString(err)
                      << " (grid=(" << blocks_seq << "," << num_kv_heads << "," << batch_size
                      << ") block=" << threads_per_block
                      << " seq=" << seq_len << ")" << std::endl;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE BACKWARD Launch Function (inverse rotation for gradients)
// ═══════════════════════════════════════════════════════════════════════════

void launchRoPERotationGQA_backward(
    float* grad_Q,
    float* grad_K,
    const float* inv_freq,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream,
    int pos_offset
) {
    // ISSUE rgb(9, 255, 0) FIX: The caller (RoPEGradFn in TensorContract_GPU.cu) intentionally
    // passes nullptr for one of grad_Q or grad_K because Q and K have independent
    // gradient paths in the autograd system. We should allow processing either one
    // individually. The validation should only fail if BOTH are null.
    // Rule 20: Crash loud on invalid inputs
    if (grad_Q == nullptr && grad_K == nullptr) {
        throw std::runtime_error(std::string(kTag) + " Both grad_Q and grad_K are null in launchRoPERotationGQA_backward");
    }
    if (inv_freq == nullptr) {
        throw std::runtime_error(std::string(kTag) + " inv_freq is null in launchRoPERotationGQA_backward");
    }
    
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        throw std::runtime_error(std::string(kTag) + " Invalid rotary_dim=" + std::to_string(rotary_dim)
            + " (head_dim=" + std::to_string(head_dim) + ")");
    }
    
    // Validate GQA configuration: num_q_heads must be divisible by num_kv_heads
    if (num_q_heads <= 0 || num_kv_heads <= 0 || (num_q_heads % num_kv_heads) != 0) {
        throw std::runtime_error(std::string(kTag) + " Invalid GQA config - num_q_heads=" + std::to_string(num_q_heads)
            + " must be divisible by num_kv_heads=" + std::to_string(num_kv_heads));
    }
    
    const int threads_per_block = 256;
    const int blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    // Drain stale CUDA errors (same rationale as forward launch)
    {
        cudaError_t stale = cudaGetLastError();
        if (stale != cudaSuccess) {
            std::cerr << kTag << " WARNING: drained stale CUDA error before RoPE backward: "
                      << cudaGetErrorString(stale)
                      << " (code=" << static_cast<int>(stale) << ")" << std::endl;
        }
    }
    
    // ISSUE #119 FIX: Only launch kernel if the corresponding gradient pointer is valid.
    // The caller passes nullptr for one of grad_Q/grad_K to process them separately.
    // The kernel uses is_q_pass to select which pointer to dereference, so the other
    // pointer is never accessed, but we still guard the launch for clarity and safety.
    
    // Launch for grad_Q (with num_q_heads) - inverse rotation
    if (grad_Q) {
        dim3 grid(blocks_seq, num_q_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQABackwardKernel<<<grid, block, 0, stream>>>(
            grad_Q, grad_K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            true,  // grad_Q pass
            pos_offset
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << kTag << " grad_Q backward kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    
    // Launch for grad_K (with num_kv_heads) - inverse rotation
    if (grad_K) {
        dim3 grid(blocks_seq, num_kv_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQABackwardKernel<<<grid, block, 0, stream>>>(
            grad_Q, grad_K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            false,  // grad_K pass
            pos_offset
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << kTag << " grad_K backward kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

} // namespace GRIM::PBM
