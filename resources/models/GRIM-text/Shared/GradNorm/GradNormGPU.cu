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
 * GradNormScratch destructor  — releases all GPU + pinned-host memory (RAII)
 * 
 * FINALIZATION:
 * =============
 * After D2H of per-group sum-of-squares, CPU loop computes:
 *   - Per-type sum_sq and element count aggregation
 *   - NaN/Inf detection with first-offender tracking
 * No GPU kernel needed — data is small, trivial on CPU.
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
constexpr int kExpectedParamGroupTypes = 9;

constexpr bool isPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

static_assert(static_cast<int>(GRIM::ParamGroupType::COUNT) == kExpectedParamGroupTypes,
              "GradNorm ParamGroupType cardinality must match accumulateGroupMetrics");
static_assert(kBlockSize >= 64, "GradNorm reduction requires kBlockSize >= 64");
static_assert(isPowerOfTwo(kBlockSize), "GradNorm reduction requires power-of-two kBlockSize");

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
    
    // Warp reduction with explicit shuffle sync.
    if (tid < 32) {
        local_sum = shared[tid] + shared[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }
    
    // Thread 0 does atomic add to output
    if (tid == 0) {
        atomicAdd(partial_sum, local_sum);
    }
}

GradNormStatus accumulateGroupMetrics(
    GradMetrics& m,
    GRIM::ParamGroupType type,
    double sum_sq,
    uint64_t count
) {
    switch (type) {
        case GRIM::ParamGroupType::EMBEDDING:
            m.embedding_sum_sq += sum_sq;
            m.embedding_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::LM_HEAD:
            m.lm_head_sum_sq += sum_sq;
            m.lm_head_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::ATTENTION:
            m.attention_sum_sq += sum_sq;
            m.attention_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::FFN:
            m.ffn_sum_sq += sum_sq;
            m.ffn_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::RMSNORM:
            m.rmsnorm_sum_sq += sum_sq;
            m.rmsnorm_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::NUMBER_ENCODER:
            m.number_encoder_sum_sq += sum_sq;
            m.number_encoder_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::ARG_SELECTOR:
            m.arg_selector_sum_sq += sum_sq;
            m.arg_selector_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::SLOT_SEED_ENCODER:
            m.slot_seed_encoder_sum_sq += sum_sq;
            m.slot_seed_encoder_count += count;
            return GradNormStatus::SUCCESS;
        case GRIM::ParamGroupType::COUNT:
            return GradNormStatus::INVALID_PARAM;
        default:
            return GradNormStatus::INVALID_PARAM;
    }
}

} // anonymous namespace

//=============================================================================
// FREE FUNCTION IMPLEMENTATIONS
//=============================================================================

GradNormScratch::~GradNormScratch() {
    if (d2h_complete_event) { cudaEventDestroy(d2h_complete_event); d2h_complete_event = nullptr; }
    if (d_partial_sums) { cudaFree(d_partial_sums); d_partial_sums = nullptr; }
    if (h_partial_sums) { cudaFreeHost(h_partial_sums); h_partial_sums = nullptr; }
    if (h_metrics) { cudaFreeHost(h_metrics); h_metrics = nullptr; }
    max_groups = 0;
    d2h_event_recorded = false;
}

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
        delete s; return nullptr;
    }

    // Pinned host: finalized metrics (written by CPU after D2H)
    err = cudaMallocHost(&s->h_metrics, sizeof(GradMetrics));
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaMallocHost h_metrics failed: %s\n", cudaGetErrorString(err));
        delete s; return nullptr;
    }

    err = cudaEventCreateWithFlags(&s->d2h_complete_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaEventCreateWithFlags d2h_complete_event failed: %s\n", cudaGetErrorString(err));
        delete s; return nullptr;
    }

    *s->h_metrics = GradMetrics{};
    return s;
}

GradNormStatus measureGradientNormsLaunch(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch,
    cudaStream_t stream
) {
    StreamController::fatalIfDefaultStream(stream, "measureGradientNormsLaunch");
    if (!scratch || scratch->max_groups == 0) {
        return GradNormStatus::NOT_INITIALIZED;
    }
    if (!scratch->d_partial_sums || !scratch->h_partial_sums || !scratch->h_metrics || !scratch->d2h_complete_event) {
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
    scratch->d2h_event_recorded = false;

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
        blocks = std::min(blocks, kMaxBlocksPerGroup);

        sumSquaresBlockKernel<<<blocks, kBlockSize, 0, stream>>>(
            grads, sz, &scratch->d_partial_sums[g]
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: sumSquaresBlockKernel launch failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::KERNEL_LAUNCH_FAILED;
    }

    // Phase 2: D2H copy of per-group sum-of-squares (caller must sync and then call Finalize)
    err = cudaMemcpyAsync(scratch->h_partial_sums, scratch->d_partial_sums,
                          num_groups * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: async D2H copy failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }

    err = cudaEventRecord(scratch->d2h_complete_event, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaEventRecord d2h_complete_event failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }
    scratch->d2h_event_recorded = true;

    return GradNormStatus::SUCCESS;
}

GradNormStatus measureGradientNormsFinalize(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch
) {
    if (!scratch || !scratch->h_partial_sums || !scratch->h_metrics || !scratch->d2h_complete_event) {
        return GradNormStatus::NOT_INITIALIZED;
    }
    if (!groups || num_groups == 0) {
        return GradNormStatus::INVALID_PARAM;
    }
    if (!scratch->d2h_event_recorded) {
        fprintf(stderr, "[GradNorm] ERROR: finalize called before a D2H completion event was recorded\n");
        return GradNormStatus::INVALID_PARAM;
    }

    cudaError_t err = cudaEventSynchronize(scratch->d2h_complete_event);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaEventSynchronize d2h_complete_event failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }

    // Phase 4: CPU finalization — per-type aggregation, NaN/Inf detection
    GradMetrics& m = *scratch->h_metrics;
    m = GradMetrics{};  // Zero everything

    for (size_t g = 0; g < num_groups; ++g) {
        if (!groups[g].grads() || groups[g].size() == 0) {
            continue;
        }

        float sq = scratch->h_partial_sums[g];
        size_t sz = groups[g].size();

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

        const GradNormStatus type_status = accumulateGroupMetrics(
            m,
            groups[g].type,
            static_cast<double>(sq),
            static_cast<uint64_t>(sz));
        if (type_status != GradNormStatus::SUCCESS) {
            fprintf(stderr, "[GradNorm] ERROR: invalid ParamGroupType=%d at group=%zu\n",
                    static_cast<int>(groups[g].type), g);
            return type_status;
        }
    }

    // Aggregate metrics
    m.groups_processed = static_cast<uint32_t>(num_groups);

    if (m.has_nan) return GradNormStatus::NAN_DETECTED;
    if (m.has_inf) return GradNormStatus::INF_DETECTED;
    return GradNormStatus::SUCCESS;
}

GradNormStatus measureGradientNorms(
    const GRIM::ParameterGroup* groups,
    size_t num_groups,
    GradNormScratch* scratch,
    cudaStream_t stream
) {
    GradNormStatus st = measureGradientNormsLaunch(groups, num_groups, scratch, stream);
    if (st != GradNormStatus::SUCCESS) {
        return st;
    }
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GradNorm] FATAL: cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        return GradNormStatus::CUDA_ERROR;
    }
    return measureGradientNormsFinalize(groups, num_groups, scratch);
}


} // namespace GRIM::GradNorm
