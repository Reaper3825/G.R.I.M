#define USE_CUDA

#include "AttentionDiagnostics.hpp"
#include <cstring>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace GRIM {
namespace {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_float = __int_as_float(expected);
        if (value <= old_float) return old_float;
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (expected != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_float = __int_as_float(expected);
        if (value >= old_float) return old_float;
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (expected != old);
    return __int_as_float(old);
}

__global__ void computeKTensorStatsKernel(
    const float* __restrict__ K,
    float* __restrict__ stats_output,
    int total_elements) {
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_sum_sq;
    __shared__ int s_nan_count;
    __shared__ int s_inf_count;

    if (threadIdx.x == 0) {
        s_min = INFINITY;
        s_max = -INFINITY;
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
        s_nan_count = 0;
        s_inf_count = 0;
    }
    __syncthreads();

    float local_min = INFINITY;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int local_nan = 0;
    int local_inf = 0;

    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        float val = K[i];
        if (isnan(val)) {
            local_nan++;
        } else if (isinf(val)) {
            local_inf++;
        } else {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    atomicMinFloat(&s_min, local_min);
    atomicMaxFloat(&s_max, local_max);
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    atomicAdd(&s_nan_count, local_nan);
    atomicAdd(&s_inf_count, local_inf);
    __syncthreads();

    if (threadIdx.x == 0) {
        stats_output[0] = s_min;
        stats_output[1] = s_max;
        stats_output[2] = s_sum;
        stats_output[3] = s_sum_sq;
        stats_output[4] = static_cast<float>(s_nan_count);
        stats_output[5] = static_cast<float>(s_inf_count);
    }
}

} // namespace

namespace {

struct KTracePayload {
    float* h_stats = nullptr;
    float* d_stats = nullptr;
    int total_elements = 0;
    int layer_idx = 0;
    int step = 0;
    char* op_name = nullptr;
};

void CUDART_CB KTraceHostCallback(void* user_data) {
    auto* payload = static_cast<KTracePayload*>(user_data);
    if (!payload || !payload->h_stats) {
        std::abort();
    }

    KTensorTrace& trace = getKTensorTrace();
    trace.k_min = payload->h_stats[0];
    trace.k_max = payload->h_stats[1];
    const float sum = payload->h_stats[2];
    const float sum_sq = payload->h_stats[3];
    trace.k_nan_count = static_cast<int>(payload->h_stats[4]);
    trace.k_inf_count = static_cast<int>(payload->h_stats[5]);

    const int valid_count = payload->total_elements - trace.k_nan_count - trace.k_inf_count;
    trace.k_mean = (valid_count > 0) ? (sum / valid_count) : 0.0f;
    const float variance = (valid_count > 0)
        ? ((sum_sq / valid_count) - (trace.k_mean * trace.k_mean))
        : 0.0f;
    trace.k_std = sqrtf(fmaxf(0.0f, variance));

    constexpr float INVALID_THRESHOLD = -1e30f;
    const bool has_flt_max = (trace.k_min < INVALID_THRESHOLD) || (trace.k_max < INVALID_THRESHOLD);

    printf("[K-TRACE] step=%d layer=%d op='%s' n=%d: min=%.6e max=%.6e mean=%.6e std=%.6e nan=%d inf=%d\n",
           payload->step, payload->layer_idx,
           payload->op_name ? payload->op_name : "<null>", payload->total_elements,
           trace.k_min, trace.k_max, trace.k_mean, trace.k_std,
           trace.k_nan_count, trace.k_inf_count);

    if (trace.k_nan_count > 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: NaN DETECTED IN K TENSOR                                 ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", payload->step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", payload->layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", payload->op_name ? payload->op_name : "<null>");
        fprintf(stderr, "║ NaN count: %d / %d                                               \n", trace.k_nan_count, payload->total_elements);
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }

    if (trace.k_inf_count > 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: Inf DETECTED IN K TENSOR                                 ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", payload->step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", payload->layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", payload->op_name ? payload->op_name : "<null>");
        fprintf(stderr, "║ Inf count: %d / %d                                               \n", trace.k_inf_count, payload->total_elements);
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }

    if (has_flt_max) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: -FLT_MAX DETECTED IN K TENSOR                            ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", payload->step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", payload->layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", payload->op_name ? payload->op_name : "<null>");
        fprintf(stderr, "║ K min:     %.6e                                                  \n", trace.k_min);
        fprintf(stderr, "║ K max:     %.6e                                                  \n", trace.k_max);
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }

    if (payload->d_stats) {
        cudaFree(payload->d_stats);
    }
    if (payload->h_stats) {
        cudaFreeHost(payload->h_stats);
    }
    delete[] payload->op_name;
    delete payload;
}

} // namespace

void traceKTensor(const float* K,
                  int total_elements,
                  const char* operation_name,
                  int layer_idx,
                  cudaStream_t stream) {
    KTensorTrace& trace = getKTensorTrace();
    if (!trace.enabled) {
        return;
    }
    if (!K || total_elements <= 0) {
        fprintf(stderr, "[K-TRACE] Invalid K tensor for op '%s' (ptr=%p elems=%d)\n",
                operation_name ? operation_name : "<null>", static_cast<const void*>(K), total_elements);
        std::abort();
    }

    float* d_stats = nullptr;
    if (cudaMalloc(&d_stats, 6 * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[K-TRACE] Failed to allocate device stats buffer\n");
        std::abort();
    }

    computeKTensorStatsKernel<<<1, 256, 0, stream>>>(K, d_stats, total_elements);

    float* h_stats = nullptr;
    if (cudaMallocHost(&h_stats, 6 * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "[K-TRACE] Failed to allocate host stats buffer\n");
        cudaFree(d_stats);
        std::abort();
    }

    cudaMemcpyAsync(h_stats, d_stats, 6 * sizeof(float), cudaMemcpyDeviceToHost, stream);

    auto* payload = new KTracePayload{};
    payload->h_stats = h_stats;
    payload->d_stats = d_stats;
    payload->total_elements = total_elements;
    payload->layer_idx = layer_idx;
    payload->step = trace.current_step;
    if (operation_name) {
        const size_t len = std::strlen(operation_name);
        payload->op_name = new char[len + 1];
        std::memcpy(payload->op_name, operation_name, len + 1);
    }

    cudaError_t cb_err = cudaLaunchHostFunc(stream, KTraceHostCallback, payload);
    if (cb_err != cudaSuccess) {
        fprintf(stderr, "[K-TRACE] Failed to enqueue host callback: %s\n", cudaGetErrorString(cb_err));
        cudaFreeHost(h_stats);
        cudaFree(d_stats);
        delete[] payload->op_name;
        delete payload;
        std::abort();
    }
}

} // namespace GRIM
