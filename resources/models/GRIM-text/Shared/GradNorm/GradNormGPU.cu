/**
 * @file GradNormGPU.cu
 * @brief GPU gradient norm measurement — kernel + free functions
 * 
 * KERNEL:
 * =======
 * sumSquaresBlockKernel — block-level sum-of-squares with shared mem reduction
 *                         One launch per parameter group, atomicAdd to per-group accumulator.
 *
 * FREE FUNCTIONS:
 * ===============
 * allocateGradNormScratch()   — one-time GPU + pinned-host buffer allocation
 * measureGradientNorms()      — launch kernel + D2H + sync + CPU finalize
 * freeGradNormScratch()       — release all memory
 * 
 * FINALIZATION:
 * =============
 * After D2H of per-group sum-of-squares, CPU loop computes:
 *   - Per-type sum_sq and element count aggregation
 *   - NaN/Inf detection with first-offender tracking
 *   - total_norm (L2 across all params)
 *   - max_group_norm
 * No GPU kernel needed — data is ~100 floats, trivial on CPU.
 */

#include "GradNormGPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace GRIM::GradNorm {

//=============================================================================
// STATUS STRING CONVERSION
//=============================================================================

const char* statusToString(GradNormStatus status) {
    switch (status) {
        case GradNormStatus::SUCCESS:              return "SUCCESS";
        case GradNormStatus::NOT_INITIALIZED:      return "NOT_INITIALIZED";
        case GradNormStatus::INVALID_PARAM:        return "INVALID_PARAM";
        case GradNormStatus::CUDA_ERROR:           return "CUDA_ERROR";
        case GradNormStatus::ALLOC_FAILED:         return "ALLOC_FAILED";
        case GradNormStatus::KERNEL_LAUNCH_FAILED: return "KERNEL_LAUNCH_FAILED";
        case GradNormStatus::NAN_DETECTED:         return "NAN_DETECTED";
        case GradNormStatus::INF_DETECTED:         return "INF_DETECTED";
        default:                                   return "UNKNOWN";
    }
}

//=============================================================================
// CUDA KERNEL
//=============================================================================

namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr int kMaxBlocksPerGroup = HyperParameters::CUDA_REDUCTION_MAX_BLOCKS;

/**
 * sumSquaresBlockKernel - Sum of squares reduction for one gradient buffer
 * 
 * Grid: (blocks_for_this_buffer)
 * Block: (kBlockSize)
 * 
 * Each block reduces its portion to shared memory, then atomicAdd to output.
 */
__global__ void sumSquaresBlockKernel(
    const float* __restrict__ grads,
    size_t size,
    float* __restrict__ partial_sum  // Atomic destination
) {
    __shared__ float shared[kBlockSize];
    
    const size_t tid = threadIdx.x;
    const size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride accumulation
    float local_sum = 0.0f;
    for (size_t i = global_idx; i < size; i += stride) {
        float val = grads[i];
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Block reduction (power-of-2 tree)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // Warp reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* vshared = shared;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }
    
    // Thread 0 does atomic add to output
    if (tid == 0) {
        atomicAdd(partial_sum, shared[0]);
    }
}

} // anonymous namespace

//=============================================================================
// FREE FUNCTION IMPLEMENTATIONS
//=============================================================================

GradNormScratch* allocateGradNormScratch(size_t max_groups, cudaStream_t stream) {
    if (max_groups == 0) {
        fprintf(stderr, "[GradNorm] FATAL: allocateGradNormScratch called with max_groups=0\n");
        return nullptr;
    }
    StreamController::fatalIfDefaultStream(stream, "allocateGradNormScratch");

    auto* s = new GradNormScratch{};
    s->max_groups = max_groups;
    cudaError_t err;

    // GPU: per-group sum-of-squares accumulator
    err = cudaMalloc(&s->d_partial_sums, max_groups * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMalloc d_partial_sums failed: %s\n", cudaGetErrorString(err));
        delete s; return nullptr;
    }

    // Pinned host: D2H target for per-group sum-of-squares
    err = cudaMallocHost(&s->h_partial_sums, max_groups * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMallocHost h_partial_sums failed: %s\n", cudaGetErrorString(err));
        cudaFree(s->d_partial_sums); delete s; return nullptr;
    }

    // Pinned host: finalized metrics (written by CPU after D2H)
    err = cudaMallocHost(&s->h_metrics, sizeof(GradMetrics));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMallocHost h_metrics failed: %s\n", cudaGetErrorString(err));
        cudaFree(s->d_partial_sums); cudaFreeHost(s->h_partial_sums); delete s; return nullptr;
    }

    *s->h_metrics = GradMetrics{};
    return s;
}

GradNormStatus measureGradientNorms(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch,
    cudaStream_t stream
) {
    if (!scratch || scratch->max_groups == 0) {
        return GradNormStatus::NOT_INITIALIZED;
    }
    if (!groups || num_groups == 0) {
        return GradNormStatus::INVALID_PARAM;
    }
    if (num_groups > scratch->max_groups) {
        fprintf(stderr, "[GradNorm] ERROR: num_groups (%zu) > max_groups (%zu)\n",
                num_groups, scratch->max_groups);
        return GradNormStatus::INVALID_PARAM;
    }

    cudaError_t err;

    // Phase 0: Zero per-group accumulators
    err = cudaMemsetAsync(scratch->d_partial_sums, 0, num_groups * sizeof(float), stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMemsetAsync d_partial_sums failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }

    // Phase 1: Sum of squares per group (one kernel launch per group)
    for (size_t g = 0; g < num_groups; ++g) {
        float* grads = groups[g].grads();
        size_t sz = groups[g].size();
        if (!grads || sz == 0) continue;

        int blocks = static_cast<int>((sz + kBlockSize - 1) / kBlockSize);
        blocks = min(blocks, kMaxBlocksPerGroup);

        sumSquaresBlockKernel<<<blocks, kBlockSize, 0, stream>>>(
            grads, sz, &scratch->d_partial_sums[g]
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: sumSquaresBlockKernel launch failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::KERNEL_LAUNCH_FAILED;
    }

    // Phase 2: D2H copy of per-group sum-of-squares
    err = cudaMemcpyAsync(scratch->h_partial_sums, scratch->d_partial_sums,
                          num_groups * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: async D2H copy failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }

    // Phase 3: Sync stream (wait for all GPU work + D2H to complete)
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }

    // Phase 4: CPU finalization — per-type aggregation, NaN/Inf detection
    GradMetrics& m = *scratch->h_metrics;
    m = GradMetrics{};  // Zero everything

    // Per-type accumulators (indexed by ParamGroupType enum: 0..5)
    float type_sum_sq[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int type_count[6] = {0, 0, 0, 0, 0, 0};
    float total_sum_sq = 0.0f;
    float max_group_norm = 0.0f;

    for (size_t g = 0; g < num_groups; ++g) {
        float sq = scratch->h_partial_sums[g];
        size_t sz = groups[g].size();
        int type_idx = static_cast<int>(groups[g].type);

        // NaN/Inf detection
        if (std::isnan(sq)) {
            if (!m.has_nan) {
                m.has_nan = 1;
                m.first_nan_group = static_cast<int32_t>(g);
                m.first_nan_value = sq;
            }
            continue;  // Don't accumulate NaN into totals
        }
        if (std::isinf(sq)) {
            if (!m.has_inf) {
                m.has_inf = 1;
                m.first_inf_group = static_cast<int32_t>(g);
                m.first_inf_value = sq;
            }
            continue;  // Don't accumulate Inf into totals
        }

        // Skip empty groups
        if (sz == 0) continue;

        // Accumulate per-type
        if (type_idx >= 0 && type_idx < 6) {
            type_sum_sq[type_idx] += sq;
            type_count[type_idx] += static_cast<int>(sz);
        }

        // Track total and max
        total_sum_sq += sq;
        float group_norm = std::sqrt(sq);
        max_group_norm = std::max(max_group_norm, group_norm);
    }

    // Write per-type metrics
    m.embedding_sum_sq = type_sum_sq[0];      m.embedding_count = type_count[0];
    m.lm_head_sum_sq = type_sum_sq[1];        m.lm_head_count = type_count[1];
    m.attention_sum_sq = type_sum_sq[2];       m.attention_count = type_count[2];
    m.ffn_sum_sq = type_sum_sq[3];            m.ffn_count = type_count[3];
    m.rmsnorm_sum_sq = type_sum_sq[4];        m.rmsnorm_count = type_count[4];
    m.scratchblock_sum_sq = type_sum_sq[5];   m.scratchblock_count = type_count[5];

    // Aggregate metrics
    m.total_norm = std::sqrt(total_sum_sq);
    m.max_group_norm = max_group_norm;
    m.groups_processed = static_cast<uint32_t>(num_groups);

    return GradNormStatus::SUCCESS;
}

void freeGradNormScratch(GradNormScratch*& scratch) {
    if (!scratch) return;
    if (scratch->d_partial_sums) cudaFree(scratch->d_partial_sums);
    if (scratch->h_partial_sums) cudaFreeHost(scratch->h_partial_sums);
    if (scratch->h_metrics) cudaFreeHost(scratch->h_metrics);
    delete scratch;
    scratch = nullptr;
}

} // namespace GRIM::GradNorm
