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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

void requirePBMResourcesReady(const PBMState& state, const char* caller) {
    if (!state.initialized) {
        throw std::runtime_error(std::string(caller) + ": PBM state is not initialized");
    }
    if (!state.alibi_slopes || !state.rope_inv_freq || !state.upload_event) {
        throw std::runtime_error(std::string(caller) +
                                 ": initialized PBM state has NULL resource pointer/event");
    }
}

void logPBMTablePreview(const PBMConstructionHP& hp,
                        const PBMRuntimeOptions& runtime) {
    if (!runtime.verbose) {
        return;
    }

    std::cout << kTag << " PBM tables sourced from HyperParameters-derived immutable views:" << std::endl;
    std::cout << "    alibi_slopes=" << hp.alibi_slopes.size
              << ", rope_inv_freq=" << hp.rope_inv_freq.size << std::endl;

    const int slope_preview = std::min(hp.num_heads, 4);
    for (int h = 0; h < slope_preview; ++h) {
        std::cout << "    head[" << h << "] slope=" << hp.alibi_slopes.data[h] << std::endl;
    }
    if (hp.num_heads > slope_preview) {
        std::cout << "    ... (" << (hp.num_heads - slope_preview) << " more slopes)" << std::endl;
    }

    const int rope_preview = std::min(static_cast<int>(hp.rope_inv_freq.size), 4);
    for (int i = 0; i < rope_preview; ++i) {
        std::cout << "    inv_freq[" << i << "]=" << hp.rope_inv_freq.data[i] << std::endl;
    }
    if (static_cast<int>(hp.rope_inv_freq.size) > rope_preview) {
        std::cout << "    ... (" << (static_cast<int>(hp.rope_inv_freq.size) - rope_preview)
                  << " more inverse frequencies)" << std::endl;
    }
}

void requireRoPELaunchGeometry(const GRIM::Batching::BatchPayload& payload,
                               const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
                               int rotary_dim,
                               const char* caller) {
    payload.validate(caller);
    if (hp.d_model <= 0 || hp.num_heads <= 0 || hp.num_kv_heads <= 0 || hp.head_dim <= 0) {
        throw std::runtime_error(std::string(kTag) + " " + caller +
            ": invalid attention dimensions d_model=" + std::to_string(hp.d_model) +
            " num_heads=" + std::to_string(hp.num_heads) +
            " num_kv_heads=" + std::to_string(hp.num_kv_heads) +
            " head_dim=" + std::to_string(hp.head_dim));
    }

    if (rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > hp.head_dim) {
        throw std::runtime_error(std::string(kTag) + " " + caller +
            ": invalid rotary_dim=" + std::to_string(rotary_dim) +
            " (head_dim=" + std::to_string(hp.head_dim) + ")");
    }
    if (hp.num_heads <= 0 || hp.num_kv_heads <= 0 ||
        (hp.num_heads % hp.num_kv_heads) != 0) {
        throw std::runtime_error(std::string(kTag) + " " + caller +
            ": invalid GQA config - num_heads=" + std::to_string(hp.num_heads) +
            " must be divisible by num_kv_heads=" + std::to_string(hp.num_kv_heads));
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
//  Core API Implementation
// ═══════════════════════════════════════════════════════════════════════════

bool initializePBM(const PBMConstructionHP& hp,
                   PBMState& state,
                   PBMRuntimeOptions runtime) {
    GRIM::PBM::requirePBMComputedTables(hp, "PBM::initializePBM");
    // Clean up any existing state
    releasePBM(state);
    
    std::cout << kTag << " Initializing Hybrid ALiBi+RoPE..." << std::endl;
    logPBMTablePreview(hp, runtime);
    
    const size_t alibi_bytes = hp.alibi_slopes.size * sizeof(float);
    const size_t rope_bytes = hp.rope_inv_freq.size * sizeof(float);
    
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
    if (runtime.stream) {
        // Async upload
        if (!checkCuda(cudaMemcpyAsync(state.alibi_slopes, hp.alibi_slopes.data,
                                        alibi_bytes, cudaMemcpyHostToDevice, runtime.stream),
                       "cudaMemcpyAsync(alibi_slopes)")) {
            releasePBM(state);
            return false;
        }
        if (!checkCuda(cudaMemcpyAsync(state.rope_inv_freq, hp.rope_inv_freq.data,
                                        rope_bytes, cudaMemcpyHostToDevice, runtime.stream),
                       "cudaMemcpyAsync(rope_inv_freq)")) {
            releasePBM(state);
            return false;
        }
    } else {
        // Sync upload
        if (!checkCuda(cudaMemcpy(state.alibi_slopes, hp.alibi_slopes.data,
                                   alibi_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(alibi_slopes)")) {
            releasePBM(state);
            return false;
        }
        if (!checkCuda(cudaMemcpy(state.rope_inv_freq, hp.rope_inv_freq.data,
                                   rope_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(rope_inv_freq)")) {
            releasePBM(state);
            return false;
        }
    }
    
    // Record event after async upload for cross-stream safety
    if (runtime.stream) {
        if (state.upload_event == nullptr) {
            if (!checkCuda(cudaEventCreateWithFlags(&state.upload_event, cudaEventDisableTiming),
                           "cudaEventCreate(upload_event)")) {
                releasePBM(state);
                return false;
            }
        }
        if (!checkCuda(cudaEventRecord(state.upload_event, runtime.stream),
                       "cudaEventRecord(upload_event)")) {
            releasePBM(state);
            return false;
        }
    }
    
    state.initialized = true;
    
    std::cout << kTag << " ✓ Hybrid ALiBi+RoPE initialized successfully" << std::endl;
    std::cout << "    ALiBi: " << hp.num_heads << " heads, slopes @ "
              << (void*)state.alibi_slopes << std::endl;
    std::cout << "    RoPE:  rotary_dim=" << hp.rotary_dim
              << ", inv_freq @ " << (void*)state.rope_inv_freq << std::endl;
    
    return true;
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

    state.initialized = false;
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
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
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

    requireRoPELaunchGeometry(payload, hp, rotary_dim, "PBM::launchRoPERotationGQA");

    const int batch_size = payload.batch_size;
    const int seq_len = payload.max_seq_len;
    const int num_q_heads = hp.num_heads;
    const int num_kv_heads = hp.num_kv_heads;
    const int head_dim = hp.head_dim;
    
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
            throw std::runtime_error(std::string(kTag) + " Q rotation kernel launch error: " +
                cudaGetErrorString(err) +
                " (grid=(" + std::to_string(blocks_seq) + "," +
                std::to_string(num_q_heads) + "," + std::to_string(batch_size) +
                ") block=" + std::to_string(threads_per_block) +
                " seq=" + std::to_string(seq_len) +
                " head_dim=" + std::to_string(head_dim) +
                " rotary=" + std::to_string(rotary_dim) + ")");
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
            throw std::runtime_error(std::string(kTag) + " K rotation kernel launch error: " +
                cudaGetErrorString(err) +
                " (grid=(" + std::to_string(blocks_seq) + "," +
                std::to_string(num_kv_heads) + "," + std::to_string(batch_size) +
                ") block=" + std::to_string(threads_per_block) +
                " seq=" + std::to_string(seq_len) + ")");
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
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
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

    requireRoPELaunchGeometry(payload, hp, rotary_dim, "PBM::launchRoPERotationGQA_backward");

    const int batch_size = payload.batch_size;
    const int seq_len = payload.max_seq_len;
    const int num_q_heads = hp.num_heads;
    const int num_kv_heads = hp.num_kv_heads;
    const int head_dim = hp.head_dim;
    
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
            throw std::runtime_error(std::string(kTag) +
                " grad_Q backward kernel launch error: " + cudaGetErrorString(err));
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
            throw std::runtime_error(std::string(kTag) +
                " grad_K backward kernel launch error: " + cudaGetErrorString(err));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RoPE cos/sin table builder (FlashAttention-v2 fused-rotary decode path)
//
//  FA2's KV-cache kernel applies RoPE from precomputed cos/sin tables rather
//  than inv_freq. Each (pos, j) element is cos/sin(pos * inv_freq[j]); this is
//  exactly the angle ropeRotationGQAKernel uses (theta = pos * inv_freq[pair]),
//  and the interleaved (2j, 2j+1) pairing matches FA2's copy_rotary_interleaved,
//  so the fused decode rotation is numerically identical to training RoPE.
// ═══════════════════════════════════════════════════════════════════════════
template<typename ElemT>
__global__ void buildRotaryCosSinTablesKernel(
    const float* __restrict__ inv_freq,
    ElemT* __restrict__ cos_out,
    ElemT* __restrict__ sin_out,
    int seqlen,
    int num_pairs
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seqlen * num_pairs;
    if (idx >= total) return;

    const int pos = idx / num_pairs;
    const int j = idx - pos * num_pairs;
    const float theta = static_cast<float>(pos) * inv_freq[j];
    cos_out[idx] = static_cast<ElemT>(cosf(theta));
    sin_out[idx] = static_cast<ElemT>(sinf(theta));
}

void launchBuildRotaryCosSinTables(
    const float* inv_freq,
    void* cos_out,
    void* sin_out,
    int seqlen,
    int rotary_dim,
    bool is_bf16,
    cudaStream_t stream
) {
    // Rule 20: crash loud on invalid inputs.
    if (!inv_freq) {
        throw std::runtime_error(std::string(kTag) + " launchBuildRotaryCosSinTables: inv_freq is null");
    }
    if (!cos_out || !sin_out) {
        throw std::runtime_error(std::string(kTag) + " launchBuildRotaryCosSinTables: cos_out/sin_out is null");
    }
    if (seqlen <= 0) {
        throw std::runtime_error(std::string(kTag) + " launchBuildRotaryCosSinTables: seqlen must be > 0, got " +
                                 std::to_string(seqlen));
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::runtime_error(std::string(kTag) + " launchBuildRotaryCosSinTables: rotary_dim must be positive and even, got " +
                                 std::to_string(rotary_dim));
    }

    const int num_pairs = rotary_dim / 2;
    const int total = seqlen * num_pairs;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (is_bf16) {
        buildRotaryCosSinTablesKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            inv_freq,
            reinterpret_cast<__nv_bfloat16*>(cos_out),
            reinterpret_cast<__nv_bfloat16*>(sin_out),
            seqlen, num_pairs);
    } else {
        buildRotaryCosSinTablesKernel<half><<<blocks, threads, 0, stream>>>(
            inv_freq,
            reinterpret_cast<half*>(cos_out),
            reinterpret_cast<half*>(sin_out),
            seqlen, num_pairs);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(kTag) +
            " launchBuildRotaryCosSinTables launch error: " + cudaGetErrorString(err));
    }
}

} // namespace GRIM::PBM
