#pragma once
/**
 * @file GradNormGPU.hpp
 * @brief GPU gradient norm measurement — free functions + scratch struct
 * 
 * DESIGN:
 * =======
 * Plain struct (GradNormScratch) holds pre-allocated GPU/host buffers.
 * Three free functions operate on it:
 *   allocateGradNormScratch()  — one-time setup
 *   measureGradientNorms()     — launches GPU sum-of-squares kernel, D2H copy,
 *                                 syncs stream, finalizes metrics on CPU
 *   freeGradNormScratch()      — release GPU memory
 *
 * After measureGradientNorms() returns, scratch->h_metrics is immediately valid.
 * No external sync required (sync happens internally).
 * 
 * KERNEL:
 * =======
 * sumSquaresBlockKernel — parallel reduction per parameter group (one launch per group)
 * 
 * METRICS:
 * ========
 * GradMetrics stores per-type sum_sq and element count. Callers compute:
 *   RMS = sqrt(sum_sq / count) — for gradient clipping and logging
 *
 * FAIL LOUD:
 * ==========
 * NaN/Inf flags written to GradMetrics (caller must check + crash).
 * Kernel launch failures return error status codes.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

#include "../TensorContract/TensorContract_GPU.hpp"

namespace GRIM::GradNorm {

//=============================================================================
// STATUS CODES
//=============================================================================

enum class GradNormStatus : uint8_t {
    SUCCESS = 0,
    NOT_INITIALIZED = 1,
    INVALID_PARAM = 2,
    CUDA_ERROR = 3,
    ALLOC_FAILED = 4,
    KERNEL_LAUNCH_FAILED = 5,
    NAN_DETECTED = 6,
    INF_DETECTED = 7,
};

const char* statusToString(GradNormStatus status);

//=============================================================================
// METRICS STRUCTURE (computed on CPU after D2H of per-group sum-of-squares)
//=============================================================================

struct alignas(64) GradMetrics {
    // Per-type gradient sum-of-squares (indices match ParamGroupType enum)
    float embedding_sum_sq = 0.0f;      // EMBEDDING = 0
    float lm_head_sum_sq = 0.0f;        // LM_HEAD = 1
    float attention_sum_sq = 0.0f;      // ATTENTION = 2
    float ffn_sum_sq = 0.0f;            // FFN = 3
    float rmsnorm_sum_sq = 0.0f;        // RMSNORM = 4
    float scratchblock_sum_sq = 0.0f;   // SCRATCHBLOCK = 5
    
    // Per-type element counts (for RMS computation)
    int embedding_count = 0;
    int lm_head_count = 0;
    int attention_count = 0;
    int ffn_count = 0;
    int rmsnorm_count = 0;
    int scratchblock_count = 0;
    
    uint32_t has_nan = 0;
    uint32_t has_inf = 0;
    uint32_t groups_processed = 0;
    uint32_t _pad = 0;

    int32_t first_nan_group = -1;
    int32_t first_inf_group = -1;
    float first_nan_value = 0.0f;
    float first_inf_value = 0.0f;
    
    // --- Accessor helpers ---
    
    /// RMS for a type: sqrt(sum_sq / count)
    static float rms(float sum_sq, int count) {
        return count > 0 ? std::sqrt(sum_sq / static_cast<float>(count)) : 0.0f;
    }
};

//=============================================================================
// SCRATCH BUFFERS (pre-allocated GPU + pinned host memory)
//=============================================================================

struct GradNormScratch {
    float* d_partial_sums = nullptr;     // [max_groups] GPU: per-group sum-of-squares accumulator
    float* h_partial_sums = nullptr;     // [max_groups] pinned host: D2H of per-group sum-of-squares
    GradMetrics* h_metrics = nullptr;    // pinned host metrics (valid after measureGradientNorms returns)
    size_t max_groups = 0;
};

//=============================================================================
// FREE FUNCTIONS
//=============================================================================

/**
 * Allocate GPU + pinned-host buffers for gradient norm measurement.
 * Call once during training setup. Returns null on failure (logged).
 */
GradNormScratch* allocateGradNormScratch(size_t max_groups, cudaStream_t stream);

/**
 * Measure gradient norms for all parameter groups.
 * Launches GPU sum-of-squares kernel, async D2H copy, syncs stream,
 * then finalizes metrics on CPU (per-type aggregation, NaN/Inf detection).
 * 
 * After return, scratch->h_metrics is immediately valid — no external sync needed.
 */
GradNormStatus measureGradientNorms(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch,
    cudaStream_t stream
);

/**
 * Free all GPU + pinned-host memory owned by the scratch struct.
 * Sets the pointer to nullptr.
 */
void freeGradNormScratch(GradNormScratch*& scratch);

} // namespace GRIM::GradNorm
