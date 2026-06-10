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
 * NaN/Inf flags written to GradMetrics and returned as hard status codes.
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
    // Per-type gradient sum-of-squares. Do not index these by enum ordinal;
    // GradNorm finalization maps ParamGroupType explicitly with a switch.
    double embedding_sum_sq = 0.0;
    double lm_head_sum_sq = 0.0;
    double attention_sum_sq = 0.0;
    double ffn_sum_sq = 0.0;
    double rmsnorm_sum_sq = 0.0;
    double scratchblock_sum_sq = 0.0;
    double mtp_sum_sq = 0.0;
    double execution_block_sum_sq = 0.0;

    // Per-type element counts (for RMS computation)
    uint64_t embedding_count = 0;
    uint64_t lm_head_count = 0;
    uint64_t attention_count = 0;
    uint64_t ffn_count = 0;
    uint64_t rmsnorm_count = 0;
    uint64_t scratchblock_count = 0;
    uint64_t mtp_count = 0;
    uint64_t execution_block_count = 0;
    
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
    static float rms(double sum_sq, uint64_t count) {
        return count > 0 ? static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count))) : 0.0f;
    }
};

//=============================================================================
// SCRATCH BUFFERS (pre-allocated GPU + pinned host memory)
//=============================================================================

struct GradNormScratch {
    float* d_partial_sums = nullptr;     // [max_groups] GPU: per-group sum-of-squares accumulator
    float* h_partial_sums = nullptr;     // [max_groups] pinned host: D2H of per-group sum-of-squares
    GradMetrics* h_metrics = nullptr;    // pinned host metrics (valid after measureGradientNorms returns)
    cudaEvent_t d2h_complete_event = nullptr; // recorded after async D2H; finalize waits on it
    size_t max_groups = 0;
    bool d2h_event_recorded = false;

    GradNormScratch() = default;
    ~GradNormScratch();
    GradNormScratch(const GradNormScratch&) = delete;
    GradNormScratch& operator=(const GradNormScratch&) = delete;
    GradNormScratch(GradNormScratch&&) = delete;
    GradNormScratch& operator=(GradNormScratch&&) = delete;
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
 * Launch gradient norm kernels and async D2H copy (no sync).
 * Records scratch->d2h_complete_event after the async D2H copy.
 * Use this to overlap CPU work (e.g. logging) with GPU work.
 */
GradNormStatus measureGradientNormsLaunch(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch,
    cudaStream_t stream
);

/**
 * Finalize metrics on CPU after D2H has completed. This waits on the recorded
 * D2H event, so callers cannot accidentally read stale pinned host data.
 * Reads scratch->h_partial_sums, writes scratch->h_metrics.
 */
GradNormStatus measureGradientNormsFinalize(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch
);

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
