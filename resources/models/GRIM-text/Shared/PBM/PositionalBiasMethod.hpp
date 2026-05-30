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

#ifndef GRIM_SHARED_PBM_POSITIONALBIASMETHOD_HPP
#define GRIM_SHARED_PBM_POSITIONALBIASMETHOD_HPP
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "../Batching/BatchPayload.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM::PBM {

// ═══════════════════════════════════════════════════════════════════════════
//  Runtime options
// ═══════════════════════════════════════════════════════════════════════════

using PBMConstructionHP = GRIM::HyperParameters::PBMConstructionHP;

struct PBMRuntimeOptions {
    cudaStream_t stream = nullptr;
    bool verbose = false;
};

// ═══════════════════════════════════════════════════════════════════════════
//  State (GPU buffers)
// ═══════════════════════════════════════════════════════════════════════════

struct PBMState {
    // ALiBi / RoPE device tables uploaded from HyperParameters-owned derived host tables.
    float* alibi_slopes = nullptr;       // Device: [num_heads] slopes
    float* rope_inv_freq = nullptr;      // Device: [rotary_dim/2] inverse frequencies
    
    // CUDA event recorded after async upload.
    // Phase1 startup seals PBM readiness before any forward pass begins; this
    // event is startup-owned upload bookkeeping, not a forward-time dependency.
    cudaEvent_t upload_event = nullptr;
    
    // Status
    bool initialized = false;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Core API
// ═══════════════════════════════════════════════════════════════════════════

// Initialize PBM state (allocates GPU buffers, computes slopes/frequencies)
bool initializePBM(const PBMConstructionHP& hp,
                   PBMState& state,
                   PBMRuntimeOptions runtime = {});

// Release GPU memory
void releasePBM(PBMState& state);

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
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
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
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
    int rotary_dim,
    cudaStream_t stream = nullptr,
    int pos_offset = 0                  // Position offset (MUST match forward pass)
);

// ═══════════════════════════════════════════════════════════════════════════
//  Memory Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline void requirePBMConstructionShape(const PBMConstructionHP& hp, const char* caller) {
    if (hp.num_heads <= 0 || hp.num_kv_heads <= 0 || hp.head_dim <= 0 || hp.rotary_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": invalid PBM construction dimensions num_heads=" +
                                 std::to_string(hp.num_heads) + " num_kv_heads=" +
                                 std::to_string(hp.num_kv_heads) + " head_dim=" +
                                 std::to_string(hp.head_dim) + " rotary_dim=" +
                                 std::to_string(hp.rotary_dim));
    }
    if ((hp.rotary_dim & 1) != 0 || hp.rotary_dim > hp.head_dim) {
        throw std::runtime_error(std::string(caller) + ": invalid PBM rotary_dim=" +
                                 std::to_string(hp.rotary_dim) + " for head_dim=" +
                                 std::to_string(hp.head_dim));
    }
}

inline void requirePBMComputedTables(const PBMConstructionHP& hp, const char* caller) {
    requirePBMConstructionShape(hp, caller);

    if (!hp.alibi_slopes.data || hp.alibi_slopes.size != static_cast<size_t>(hp.num_heads)) {
        throw std::runtime_error(std::string(caller) + ": PBM alibi_slopes view is invalid for num_heads=" +
                                 std::to_string(hp.num_heads) + " (size=" +
                                 std::to_string(hp.alibi_slopes.size) + ")");
    }

    const size_t expected_rope_size = static_cast<size_t>(hp.rotary_dim / 2);
    if (!hp.rope_inv_freq.data || hp.rope_inv_freq.size != expected_rope_size) {
        throw std::runtime_error(std::string(caller) + ": PBM rope_inv_freq view is invalid for rotary_dim=" +
                                 std::to_string(hp.rotary_dim) + " (size=" +
                                 std::to_string(hp.rope_inv_freq.size) + ")");
    }
}

// Returns GPU bytes required for PBM buffers
inline size_t getPBMDeviceBytes(const PBMConstructionHP& hp) {
    requirePBMConstructionShape(hp, "PBM::getPBMDeviceBytes");
    const size_t alibi_bytes = static_cast<size_t>(hp.num_heads) * sizeof(float);
    const size_t rope_bytes = static_cast<size_t>(hp.rotary_dim / 2) * sizeof(float);
    return alibi_bytes + rope_bytes;
}

} // namespace GRIM::PBM

#endif // GRIM_SHARED_PBM_POSITIONALBIASMETHOD_HPP
