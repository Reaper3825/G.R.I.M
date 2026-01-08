//======================================================//
//  Shared/PBM/PositionalBiasMethod.cu
//  Unified Positional Bias Method Implementation
//  
//  Combines ALiBi slope computation and RoPE inverse frequencies
//  into a single initialization path. Both are ALWAYS enabled.
//======================================================//

#define USE_CUDA

#include "PositionalBiasMethod.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace GRIM::PBM {

namespace {

constexpr const char* kTag = "[PBM]";

bool checkCuda(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::cerr << kTag << " " << what << " failed: " << cudaGetErrorString(err) << std::endl;
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ALiBi Slope Computation (Host)
//  Formula: slope[h] = 2^(exponent * (h+1) / num_heads)
//  With exponent=-8: slopes decay geometrically from 2^-1 to 2^-8
// ═══════════════════════════════════════════════════════════════════════════

bool computeAlibiSlopes(const PBMConfig& config, std::vector<float>& out_slopes) {
    if (config.num_heads <= 0) {
        std::cerr << kTag << " Invalid num_heads=" << config.num_heads << std::endl;
        return false;
    }
    
    out_slopes.resize(static_cast<size_t>(config.num_heads));
    for (int h = 0; h < config.num_heads; ++h) {
        const float exponent = config.alibi_slope_exponent * 
            static_cast<float>(h + 1) / static_cast<float>(config.num_heads);
        out_slopes[static_cast<size_t>(h)] = std::pow(2.0f, exponent);
    }
    
    if (config.verbose) {
        std::cout << kTag << " ALiBi slopes computed for " << config.num_heads << " heads:" << std::endl;
        const int preview = std::min(config.num_heads, 4);
        for (int h = 0; h < preview; ++h) {
            std::cout << "  head[" << h << "] slope=" << out_slopes[static_cast<size_t>(h)] << std::endl;
        }
        if (config.num_heads > 4) {
            std::cout << "  ... (" << (config.num_heads - 4) << " more heads)" << std::endl;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE Inverse Frequency Computation (Host)
//  Formula: inv_freq[i] = scaling / theta^(2*i / rotary_dim)
//  Standard: theta=10000, scaling=1.0, rotary_dim=head_dim
// ═══════════════════════════════════════════════════════════════════════════

bool computeRoPEInvFreq(const PBMConfig& config, std::vector<float>& out_inv_freq) {
    if (config.rotary_dim <= 0 || (config.rotary_dim & 1) != 0) {
        std::cerr << kTag << " Invalid rotary_dim=" << config.rotary_dim 
                  << " (must be positive and even)" << std::endl;
        return false;
    }
    
    const int half_dim = config.rotary_dim / 2;
    out_inv_freq.resize(static_cast<size_t>(half_dim));
    
    for (int i = 0; i < half_dim; ++i) {
        const float exp_arg = static_cast<float>(2 * i) / static_cast<float>(config.rotary_dim);
        out_inv_freq[static_cast<size_t>(i)] = config.rope_scaling / 
            std::pow(config.rope_theta, exp_arg);
    }
    
    if (config.verbose) {
        std::cout << kTag << " RoPE inverse frequencies computed:" << std::endl;
        std::cout << "  rotary_dim=" << config.rotary_dim 
                  << ", theta=" << config.rope_theta
                  << ", scaling=" << config.rope_scaling << std::endl;
        const int preview = std::min(half_dim, 4);
        for (int i = 0; i < preview; ++i) {
            std::cout << "  inv_freq[" << i << "]=" << out_inv_freq[static_cast<size_t>(i)] << std::endl;
        }
        if (half_dim > 4) {
            std::cout << "  ... (" << (half_dim - 4) << " more frequencies)" << std::endl;
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
    // Step 5: Store metadata
    // ─────────────────────────────────────────────────────────────────────────
    state.num_heads = config.num_heads;
    state.head_dim = config.head_dim;
    state.rotary_dim = config.rotary_dim;
    state.num_kv_heads = config.num_kv_heads;
    state.initialized = true;
    
    std::cout << kTag << " ✓ Hybrid ALiBi+RoPE initialized successfully" << std::endl;
    std::cout << "    ALiBi: " << state.num_heads << " heads, slopes @ " 
              << (void*)state.alibi_slopes << std::endl;
    std::cout << "    RoPE:  rotary_dim=" << state.rotary_dim 
              << ", inv_freq @ " << (void*)state.rope_inv_freq << std::endl;
    
    return true;
}

bool ensurePBM(const PBMConfig& config, PBMState& state) {
    // Check if re-initialization needed
    const bool dims_match = state.initialized &&
                            state.num_heads == config.num_heads &&
                            state.head_dim == config.head_dim &&
                            state.rotary_dim == config.rotary_dim &&
                            state.num_kv_heads == config.num_kv_heads;
    
    if (dims_match) {
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
    
    state.alibi_slopes_host.clear();
    state.alibi_slopes_host.shrink_to_fit();
    state.rope_inv_freq_host.clear();
    state.rope_inv_freq_host.shrink_to_fit();
    
    state.num_heads = 0;
    state.head_dim = 0;
    state.rotary_dim = 0;
    state.num_kv_heads = 0;
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

// Unfused RoPE kernel: Apply rotation to Q and K
// Q, K are in BHSD format: [batch, num_heads, seq_len, head_dim]
__global__ void ropeRotationKernel(
    float* __restrict__ Q,
    float* __restrict__ K,
    const float* __restrict__ inv_freq,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int rotary_dim
) {
    const int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (pos_idx >= seq_len || head_idx >= num_heads || batch_idx >= batch_size) return;
    
    // Calculate base offset for this [batch, head, position]
    const int bhsd_offset = ((batch_idx * num_heads + head_idx) * seq_len + pos_idx) * head_dim;
    
    // Apply rotation to pairs of dimensions
    const int num_pairs = rotary_dim / 2;
    for (int pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
        const int dim_i = pair_idx * 2;
        const int dim_j = pair_idx * 2 + 1;
        
        // Compute cos and sin for this position and frequency
        const float freq = inv_freq[pair_idx];
        const float theta = static_cast<float>(pos_idx) * freq;
        const float cos_val = cosf(theta);
        const float sin_val = sinf(theta);
        
        // Rotate Q
        float q_i = Q[bhsd_offset + dim_i];
        float q_j = Q[bhsd_offset + dim_j];
        applyRotation(q_i, q_j, cos_val, sin_val);
        Q[bhsd_offset + dim_i] = q_i;
        Q[bhsd_offset + dim_j] = q_j;
        
        // Rotate K
        float k_i = K[bhsd_offset + dim_i];
        float k_j = K[bhsd_offset + dim_j];
        applyRotation(k_i, k_j, cos_val, sin_val);
        K[bhsd_offset + dim_i] = k_i;
        K[bhsd_offset + dim_j] = k_j;
    }
}

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
    bool is_q_pass
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
        const float theta = static_cast<float>(pos_idx) * freq;
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
    bool is_q_pass
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
        const float theta = static_cast<float>(pos_idx) * freq;
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

void launchRoPERotation(
    float* Q,
    float* K,
    const float* inv_freq,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int rotary_dim,
    cudaStream_t stream
) {
    if (Q == nullptr || K == nullptr || inv_freq == nullptr) {
        std::cerr << kTag << " ERROR: Null pointer passed to launchRoPERotation" << std::endl;
        return;
    }
    
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        std::cerr << kTag << " ERROR: Invalid rotary_dim=" << rotary_dim 
                  << " (head_dim=" << head_dim << ")" << std::endl;
        return;
    }
    
    const int threads_per_block = 256;
    const int blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_seq, num_heads, batch_size);
    dim3 block(threads_per_block);
    
    ropeRotationKernel<<<grid, block, 0, stream>>>(
        Q, K, inv_freq,
        batch_size, num_heads, seq_len, head_dim, rotary_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << kTag << " Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

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
    cudaStream_t stream
) {
    if (Q == nullptr || K == nullptr || inv_freq == nullptr) {
        std::cerr << kTag << " ERROR: Null pointer passed to launchRoPERotationGQA" << std::endl;
        return;
    }
    
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        std::cerr << kTag << " ERROR: Invalid rotary_dim=" << rotary_dim 
                  << " (head_dim=" << head_dim << ")" << std::endl;
        return;
    }
    
    const int threads_per_block = 256;
    const int blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    // Launch for Q (with num_q_heads)
    {
        dim3 grid(blocks_seq, num_q_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQAKernel<<<grid, block, 0, stream>>>(
            Q, K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            true  // Q pass
        );
    }
    
    // Launch for K (with num_kv_heads)
    {
        dim3 grid(blocks_seq, num_kv_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQAKernel<<<grid, block, 0, stream>>>(
            Q, K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            false  // K pass
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << kTag << " GQA Kernel launch error: " << cudaGetErrorString(err) << std::endl;
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
    cudaStream_t stream
) {
    if (grad_Q == nullptr || grad_K == nullptr || inv_freq == nullptr) {
        std::cerr << kTag << " ERROR: Null pointer passed to launchRoPERotationGQA_backward" << std::endl;
        return;
    }
    
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
        std::cerr << kTag << " ERROR: Invalid rotary_dim=" << rotary_dim 
                  << " (head_dim=" << head_dim << ")" << std::endl;
        return;
    }
    
    const int threads_per_block = 256;
    const int blocks_seq = (seq_len + threads_per_block - 1) / threads_per_block;
    
    // Launch for grad_Q (with num_q_heads) - inverse rotation
    {
        dim3 grid(blocks_seq, num_q_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQABackwardKernel<<<grid, block, 0, stream>>>(
            grad_Q, grad_K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            true  // grad_Q pass
        );
    }
    
    // Launch for grad_K (with num_kv_heads) - inverse rotation
    {
        dim3 grid(blocks_seq, num_kv_heads, batch_size);
        dim3 block(threads_per_block);
        
        ropeRotationGQABackwardKernel<<<grid, block, 0, stream>>>(
            grad_Q, grad_K, inv_freq,
            batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, rotary_dim,
            false  // grad_K pass
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << kTag << " GQA Backward Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
}

} // namespace GRIM::PBM
