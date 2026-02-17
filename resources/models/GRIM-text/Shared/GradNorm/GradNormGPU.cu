/**
 * @file GradNormGPU.cu
 * @brief GPU-resident gradient norm computation - implementation
 * 
 * KERNEL ARCHITECTURE:
 * ====================
 * Phase 1: sumSquaresPerGroupKernel
 *   - Grid: (num_groups * blocks_per_group)
 *   - Each group uses multiple blocks for large buffers
 *   - Block reduction in shared memory
 *   - atomicAdd to d_partial_sums_[group_idx]
 * 
 * Phase 2: finalizeNormsKernel
 *   - Single block (256 threads)
 *   - Thread per group computes sqrt(partial_sum)
 *   - Accumulates per-type norms
 *   - Computes total_norm = sqrt(sum of squared norms)
 *   - NaN/Inf detection
 *   - Clip scale computation
 * 
 * MEMORY COALESCING:
 * ==================
 * - All gradient reads are contiguous (coalesced)
 * - Shared memory reduction avoids atomic contention
 * - Final reduction uses warp shuffle for last 32 elements
 * 
 * NO CPU SYNC:
 * ============
 * - All kernels chained on same stream
 * - Host metrics via async copy (explicit call)
 */

#include "GradNormGPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include <cstdio>
#include <cmath>
#include <atomic>

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
// CUDA KERNELS
//=============================================================================

namespace {

constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
constexpr int kMaxBlocksPerGroup = HyperParameters::CUDA_REDUCTION_MAX_BLOCKS;  // Cap grid size per group

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

/**
 * finalizeNormsKernel - Compute sqrt, per-type aggregation, flags
 * 
 * Grid: (1)
 * Block: (256)
 * 
 * Thread i handles group i (if i < num_groups).
 */
__global__ void finalizeNormsKernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ group_norms,
    const GRIM::ParamGroupType* __restrict__ types,
    int num_groups,
    float clip_threshold,
    GradMetrics* __restrict__ metrics
) {
    // ParamGroupType::COUNT = 7 (EMBEDDING, LM_HEAD, NUMERIC_HEAD, ATTENTION, FFN, RMSNORM, SCRATCHBLOCK)
    __shared__ float type_squared_sums[7];
    __shared__ float total_squared_sum;
    __shared__ float max_norm;
    __shared__ uint32_t nan_flag;
    __shared__ uint32_t inf_flag;
    __shared__ int first_nan_group;
    __shared__ int first_inf_group;
    __shared__ float first_nan_value;
    __shared__ float first_inf_value;
    
    const int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < 7) {
        type_squared_sums[tid] = 0.0f;
    }
    if (tid == 0) {
        total_squared_sum = 0.0f;
        max_norm = 0.0f;
        nan_flag = 0;
        inf_flag = 0;
        first_nan_group = -1;
        first_inf_group = -1;
        first_nan_value = 0.0f;
        first_inf_value = 0.0f;
    }
    __syncthreads();
    
    // Each thread computes sqrt for its group
    if (tid < num_groups) {
        float squared_sum = partial_sums[tid];
        float norm = sqrtf(squared_sum);
        
        // Check for NaN/Inf
        if (isnan(norm)) {
            atomicOr(&nan_flag, 1u);
            if (atomicCAS(&first_nan_group, -1, tid) == -1) {
                first_nan_value = norm;
            }
            norm = 0.0f;  // Sanitize for aggregation
        }
        if (isinf(norm)) {
            atomicOr(&inf_flag, 1u);
            if (atomicCAS(&first_inf_group, -1, tid) == -1) {
                first_inf_value = norm;
            }
            norm = 1e10f;  // Cap for aggregation
        }
        
        group_norms[tid] = norm;
        
        // Accumulate to type bucket (7 types including NUMERIC_HEAD + SCRATCHBLOCK)
        int type_idx = static_cast<int>(types[tid]);
        if (type_idx >= 0 && type_idx < 7) {
            atomicAdd(&type_squared_sums[type_idx], squared_sum);
        }
        
        // Accumulate to total
        atomicAdd(&total_squared_sum, squared_sum);
        
        // Track max norm
        atomicMax(reinterpret_cast<int*>(&max_norm), __float_as_int(norm));
    }
    
    // CRITICAL SYNC: All threads must complete atomic operations before thread 0 reads results.
    // Previous comment claimed "atomics provide memory ordering guarantees" - this is WRONG.
    // atomicAdd to shared memory does NOT guarantee visibility to other threads without a barrier.
    // Without this sync, thread 0 can read partial sums (race condition).
    __syncthreads();
    
    // Thread 0 finalizes metrics
    if (tid == 0) {
        float total = sqrtf(total_squared_sum);
        
        // Compute clip scale
        float scale = 1.0f;
        if (clip_threshold > 0.0f && total > clip_threshold) {
            scale = clip_threshold / total;
        }
        
        // Write metrics (type indices match ParamGroupType enum after NUMERIC_HEAD deletion)
        metrics->embedding_norm = sqrtf(type_squared_sums[0]);     // EMBEDDING = 0
        metrics->lm_head_norm = sqrtf(type_squared_sums[1]);       // LM_HEAD = 1
        metrics->attention_norm = sqrtf(type_squared_sums[2]);     // ATTENTION = 2
        metrics->ffn_norm = sqrtf(type_squared_sums[3]);           // FFN = 3
        metrics->rmsnorm_norm = sqrtf(type_squared_sums[4]);       // RMSNORM = 4
        metrics->scratchblock_norm = sqrtf(type_squared_sums[5]);  // SCRATCHBLOCK = 5
        metrics->total_norm = total;
        metrics->clip_scale = scale;
        metrics->max_norm = __int_as_float(atomicAdd(reinterpret_cast<int*>(&max_norm), 0));
        metrics->has_nan = nan_flag;
        metrics->has_inf = inf_flag;
        metrics->groups_processed = static_cast<uint32_t>(num_groups);
        metrics->first_nan_group = first_nan_group;
        metrics->first_inf_group = first_inf_group;
        metrics->first_nan_value = first_nan_value;
        metrics->first_inf_value = first_inf_value;
    }
}

/**
 * zeroPartialSumsKernel - Zero the partial sums buffer
 */
__global__ void zeroPartialSumsKernel(float* partial_sums, int num_groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_groups) {
        partial_sums[idx] = 0.0f;
    }
}

} // anonymous namespace

//=============================================================================
// GRAD NORM CONTROLLER IMPLEMENTATION
//=============================================================================

GradNormController::~GradNormController() {
    shutdown();
}

GradNormController::GradNormController(GradNormController&& other) noexcept
    : initialized_(other.initialized_)
    , max_groups_(other.max_groups_)
    , d_partial_sums_(other.d_partial_sums_)
    , d_group_norms_(other.d_group_norms_)
    , d_metrics_(other.d_metrics_)
    , d_types_temp_(other.d_types_temp_)
    , h_metrics_(other.h_metrics_)
    , h_types_temp_(other.h_types_temp_)
    , last_stream_(other.last_stream_)
{
    other.initialized_ = false;
    other.d_partial_sums_ = nullptr;
    other.d_group_norms_ = nullptr;
    other.d_metrics_ = nullptr;
    other.d_types_temp_ = nullptr;
    other.h_metrics_ = nullptr;
    other.h_types_temp_ = nullptr;
}

GradNormController& GradNormController::operator=(GradNormController&& other) noexcept {
    if (this != &other) {
        shutdown();
        initialized_ = other.initialized_;
        max_groups_ = other.max_groups_;
        d_partial_sums_ = other.d_partial_sums_;
        d_group_norms_ = other.d_group_norms_;
        d_metrics_ = other.d_metrics_;
        d_types_temp_ = other.d_types_temp_;
        h_metrics_ = other.h_metrics_;
        h_types_temp_ = other.h_types_temp_;
        last_stream_ = other.last_stream_;
        
        other.initialized_ = false;
        other.d_partial_sums_ = nullptr;
        other.d_group_norms_ = nullptr;
        other.d_metrics_ = nullptr;
        other.d_types_temp_ = nullptr;
        other.h_metrics_ = nullptr;
        other.h_types_temp_ = nullptr;
    }
    return *this;
}

GradNormStatus GradNormController::initialize(size_t max_groups, cudaStream_t stream) {
    if (initialized_) {
        shutdown();
    }
    
    if (max_groups == 0) {
        return GradNormStatus::INVALID_PARAM;
    }
    StreamController::fatalIfDefaultStream(stream, "GradNormController::initialize");
    
    max_groups_ = max_groups;
    last_stream_ = stream;
    
    // Allocate device memory
    cudaError_t err;
    
    err = cudaMalloc(&d_partial_sums_, max_groups_ * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMalloc d_partial_sums_ failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::ALLOC_FAILED;
    }
    
    err = cudaMalloc(&d_group_norms_, max_groups_ * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMalloc d_group_norms_ failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial_sums_);
        d_partial_sums_ = nullptr;
        return GradNormStatus::ALLOC_FAILED;
    }
    
    err = cudaMalloc(&d_metrics_, sizeof(GradMetrics));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMalloc d_metrics_ failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial_sums_);
        cudaFree(d_group_norms_);
        d_partial_sums_ = nullptr;
        d_group_norms_ = nullptr;
        return GradNormStatus::ALLOC_FAILED;
    }
    
    // PERFORMANCE FIX: Pre-allocate d_types_temp_ buffer to avoid cudaMalloc in hot path
    // OLD: cudaMalloc(&d_types, ...) every computeAndClip() call → 7+ second blocking stall
    // NEW: One-time allocation here, reused forever → zero stall
    err = cudaMalloc(&d_types_temp_, max_groups_ * sizeof(GRIM::ParamGroupType));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMalloc d_types_temp_ failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial_sums_);
        cudaFree(d_group_norms_);
        cudaFree(d_metrics_);
        d_partial_sums_ = nullptr;
        d_group_norms_ = nullptr;
        d_metrics_ = nullptr;
        return GradNormStatus::ALLOC_FAILED;
    }
    
    // Allocate pinned host buffer for types (avoid repeated malloc per batch)
    err = cudaMallocHost(&h_types_temp_, max_groups_ * sizeof(GRIM::ParamGroupType));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMallocHost h_types_temp_ failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial_sums_);
        cudaFree(d_group_norms_);
        cudaFree(d_metrics_);
        cudaFree(d_types_temp_);
        d_partial_sums_ = nullptr;
        d_group_norms_ = nullptr;
        d_metrics_ = nullptr;
        d_types_temp_ = nullptr;
        return GradNormStatus::ALLOC_FAILED;
    }
    
    // Allocate pinned host memory for async copy
    err = cudaMallocHost(&h_metrics_, sizeof(GradMetrics));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMallocHost h_metrics_ failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_partial_sums_);
        cudaFree(d_group_norms_);
        cudaFree(d_metrics_);
        cudaFree(d_types_temp_);
        d_partial_sums_ = nullptr;
        d_group_norms_ = nullptr;
        d_metrics_ = nullptr;
        d_types_temp_ = nullptr;
        return GradNormStatus::ALLOC_FAILED;
    }
    
    // Initialize device metrics to zero
    err = cudaMemsetAsync(d_metrics_, 0, sizeof(GradMetrics), stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMemsetAsync d_metrics_ failed: %s\n",
                cudaGetErrorString(err));
        shutdown();
        return GradNormStatus::CUDA_ERROR;
    }
    
    // Initialize host metrics
    *h_metrics_ = GradMetrics{};
    
    initialized_ = true;
    return GradNormStatus::SUCCESS;
}

void GradNormController::shutdown() {
    if (d_partial_sums_) {
        cudaFree(d_partial_sums_);
        d_partial_sums_ = nullptr;
    }
    if (d_group_norms_) {
        cudaFree(d_group_norms_);
        d_group_norms_ = nullptr;
    }
    if (d_metrics_) {
        cudaFree(d_metrics_);
        d_metrics_ = nullptr;
    }
    if (d_types_temp_) {
        cudaFree(d_types_temp_);
        d_types_temp_ = nullptr;
    }
    if (h_metrics_) {
        cudaFreeHost(h_metrics_);
        h_metrics_ = nullptr;
    }
    if (h_types_temp_) {
        cudaFreeHost(h_types_temp_);
        h_types_temp_ = nullptr;
    }
    initialized_ = false;
    max_groups_ = 0;
}

GradNormStatus GradNormController::computeNorms(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    cudaStream_t stream
) {
    if (!initialized_) {
        return GradNormStatus::NOT_INITIALIZED;
    }
    
    if (!groups || num_groups == 0) {
        return GradNormStatus::INVALID_PARAM;
    }
    
    if (num_groups > max_groups_) {
        fprintf(stderr, "[GradNorm] ERROR: num_groups (%zu) > max_groups_ (%zu)\n",
                num_groups, max_groups_);
        return GradNormStatus::INVALID_PARAM;
    }
    
    last_stream_ = stream;
    cudaError_t err;
    
    // Phase 0: Zero partial sums
    int zero_blocks = (static_cast<int>(num_groups) + kBlockSize - 1) / kBlockSize;
    zeroPartialSumsKernel<<<zero_blocks, kBlockSize, 0, stream>>>(
        d_partial_sums_, static_cast<int>(num_groups)
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: zeroPartialSumsKernel launch failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::KERNEL_LAUNCH_FAILED;
    }
    
    // Extract types directly to pinned host buffer (no temp allocation)
    for (size_t g = 0; g < num_groups; ++g) {
        h_types_temp_[g] = groups[g].type;
    }
    
    // Phase 1: Sum of squares for each group (extract from ParameterGroup directly)
    for (size_t g = 0; g < num_groups; ++g) {
        float* grads = groups[g].grads();
        size_t sz = groups[g].size();
        if (!grads || sz == 0) {
            continue;
        }
        
        // Compute grid size for this buffer
        int blocks = static_cast<int>((sz + kBlockSize - 1) / kBlockSize);
        blocks = min(blocks, kMaxBlocksPerGroup);
        
        sumSquaresBlockKernel<<<blocks, kBlockSize, 0, stream>>>(
            grads,
            sz,
            &d_partial_sums_[g]
        );
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: sumSquaresBlockKernel launch failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::KERNEL_LAUNCH_FAILED;
    }
    
    // Phase 2: Finalize norms
    // PERFORMANCE FIX: Reuse pre-allocated h_types_temp_ (pinned) and d_types_temp_ (device)
    err = cudaMemcpyAsync(d_types_temp_, h_types_temp_, num_groups * sizeof(GRIM::ParamGroupType),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMemcpyAsync d_types_temp_ failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }
    
    // clip_threshold=0 means no clipping - we just compute norms
    // Actual clipping is done in Phase2 via scaleGradients() after CPU-side decision
    const float clip_threshold = 0.0f;
    
    finalizeNormsKernel<<<1, kBlockSize, 0, stream>>>(
        d_partial_sums_,
        d_group_norms_,
        d_types_temp_,
        static_cast<int>(num_groups),
        clip_threshold,
        d_metrics_
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: finalizeNormsKernel launch failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::KERNEL_LAUNCH_FAILED;
    }
    
    return GradNormStatus::SUCCESS;
}

GradNormStatus GradNormController::asyncCopyToHost(cudaStream_t stream) {
    if (!initialized_) {
        return GradNormStatus::NOT_INITIALIZED;
    }
    
    cudaError_t err = cudaMemcpyAsync(
        h_metrics_,
        d_metrics_,
        sizeof(GradMetrics),
        cudaMemcpyDeviceToHost,
        stream
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: asyncCopyToHost failed: %s\n",
                cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }
    
    return GradNormStatus::SUCCESS;
}

} // namespace GRIM::GradNorm
