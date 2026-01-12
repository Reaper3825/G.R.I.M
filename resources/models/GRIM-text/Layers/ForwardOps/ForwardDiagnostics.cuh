#pragma once
/**
 * ForwardDiagnostics.cuh
 * 
 * Diagnostic utilities to trace values through the forward pass.
 * Enables logging of buffer statistics (mean, var, min, max, NaN count)
 * to debug training plateau issues.
 * 
 * EXPECTED VALUES (after proper initialization):
 * - Embeddings: mean≈0, var≈1/d_model (Xavier init)
 * - After RMSNorm: mean≈0, rms≈1 
 * - Encoder output: mean≈0, var varies by layer depth
 * - Logits: mean≈0 (if well-initialized), range [-10, 10] typical
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

namespace GRIM {
namespace Forward {

// Enable/disable all forward diagnostics (compile-time toggle)
#ifndef FORWARD_DIAGNOSTICS_ENABLED
#define FORWARD_DIAGNOSTICS_ENABLED 1
#endif

// Struct to hold buffer statistics
struct BufferStats {
    float mean;
    float variance;
    float min_val;
    float max_val;
    int nan_count;
    int inf_count;
    size_t num_elements;
    
    void print(const char* name) const {
        fprintf(stderr, "[ForwardDiag] %s: elements=%zu mean=%.6f var=%.6f min=%.4f max=%.4f nan=%d inf=%d\n",
                name, num_elements, mean, variance, min_val, max_val, nan_count, inf_count);
    }
    
    bool isHealthy() const {
        return nan_count == 0 && inf_count == 0 && 
               !std::isnan(mean) && !std::isinf(mean) &&
               !std::isnan(variance) && !std::isinf(variance);
    }
};

// Kernel to compute buffer statistics
__global__ void computeBufferStatsKernel(
    const float* __restrict__ buffer,
    size_t num_elements,
    float* __restrict__ out_sum,
    float* __restrict__ out_sq_sum,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    int* __restrict__ out_nan_count,
    int* __restrict__ out_inf_count
) {
    __shared__ float s_sum[256];
    __shared__ float s_sq_sum[256];
    __shared__ float s_min[256];
    __shared__ float s_max[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];
    
    const int tid = threadIdx.x;
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    int local_nan = 0;
    int local_inf = 0;
    
    // Grid-stride loop
    for (size_t i = gid; i < num_elements; i += stride) {
        const float val = buffer[i];
        if (isnan(val)) {
            local_nan++;
        } else if (isinf(val)) {
            local_inf++;
        } else {
            local_sum += val;
            local_sq_sum += val * val;
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Store to shared memory
    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_nan[tid] = local_nan;
    s_inf[tid] = local_inf;
    __syncthreads();
    
    // Reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            s_nan[tid] += s_nan[tid + s];
            s_inf[tid] += s_inf[tid + s];
        }
        __syncthreads();
    }
    
    // Block 0 writes result
    if (tid == 0) {
        atomicAdd(out_sum, s_sum[0]);
        atomicAdd(out_sq_sum, s_sq_sum[0]);
        
        // atomicMin/Max for float using int reinterpretation
        // Use atomic compare-and-swap for min/max
        int* min_int = (int*)out_min;
        int* max_int = (int*)out_max;
        float old_min = *out_min;
        while (s_min[0] < old_min) {
            old_min = __int_as_float(atomicCAS(min_int, __float_as_int(old_min), __float_as_int(s_min[0])));
        }
        float old_max = *out_max;
        while (s_max[0] > old_max) {
            old_max = __int_as_float(atomicCAS(max_int, __float_as_int(old_max), __float_as_int(s_max[0])));
        }
        
        atomicAdd(out_nan_count, s_nan[0]);
        atomicAdd(out_inf_count, s_inf[0]);
    }
}

// Host function to compute buffer statistics
inline BufferStats computeBufferStats(
    const float* d_buffer,
    size_t num_elements,
    cudaStream_t stream = 0
) {
    BufferStats stats = {};
    stats.num_elements = num_elements;
    
    if (!d_buffer || num_elements == 0) {
        stats.mean = 0.0f;
        stats.variance = 0.0f;
        stats.min_val = 0.0f;
        stats.max_val = 0.0f;
        return stats;
    }
    
    // Allocate device memory for reduction results
    float* d_sum;
    float* d_sq_sum;
    float* d_min;
    float* d_max;
    int* d_nan_count;
    int* d_inf_count;
    
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sq_sum, sizeof(float));
    cudaMalloc(&d_min, sizeof(float));
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_nan_count, sizeof(int));
    cudaMalloc(&d_inf_count, sizeof(int));
    
    // Initialize
    float init_sum = 0.0f;
    float init_min = FLT_MAX;
    float init_max = -FLT_MAX;
    int init_count = 0;
    cudaMemcpyAsync(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_sq_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_nan_count, &init_count, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_inf_count, &init_count, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = std::min((int)((num_elements + block_size - 1) / block_size), 1024);
    computeBufferStatsKernel<<<num_blocks, block_size, 0, stream>>>(
        d_buffer, num_elements,
        d_sum, d_sq_sum, d_min, d_max, d_nan_count, d_inf_count
    );
    
    // Copy results back
    float h_sum, h_sq_sum;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_sq_sum, d_sq_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&stats.min_val, d_min, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&stats.max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&stats.nan_count, d_nan_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&stats.inf_count, d_inf_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Compute mean and variance
    const size_t valid_count = num_elements - stats.nan_count - stats.inf_count;
    if (valid_count > 0) {
        stats.mean = h_sum / valid_count;
        stats.variance = (h_sq_sum / valid_count) - (stats.mean * stats.mean);
    }
    
    // Cleanup
    cudaFree(d_sum);
    cudaFree(d_sq_sum);
    cudaFree(d_min);
    cudaFree(d_max);
    cudaFree(d_nan_count);
    cudaFree(d_inf_count);
    
    return stats;
}

// Convenience function with expected value checking
inline void logBufferWithExpected(
    const char* name,
    const float* d_buffer,
    size_t num_elements,
    float expected_mean,
    float expected_var,
    float expected_min,
    float expected_max,
    cudaStream_t stream = 0
) {
#if FORWARD_DIAGNOSTICS_ENABLED
    BufferStats stats = computeBufferStats(d_buffer, num_elements, stream);
    
    fprintf(stderr, "[ForwardDiag] %s:\n", name);
    fprintf(stderr, "  Actual:   mean=%.6f var=%.6f min=%.4f max=%.4f (nan=%d inf=%d)\n",
            stats.mean, stats.variance, stats.min_val, stats.max_val, stats.nan_count, stats.inf_count);
    fprintf(stderr, "  Expected: mean≈%.2f var≈%.4f range=[%.2f, %.2f]\n",
            expected_mean, expected_var, expected_min, expected_max);
    
    // Flag anomalies
    if (!stats.isHealthy()) {
        fprintf(stderr, "  ❌ UNHEALTHY: Contains NaN/Inf!\n");
    }
    if (std::abs(stats.mean - expected_mean) > 1.0f) {
        fprintf(stderr, "  ⚠️  Mean deviation: %.4f (expected ≈%.2f)\n", stats.mean, expected_mean);
    }
    if (stats.variance > expected_var * 10.0f) {
        fprintf(stderr, "  ⚠️  High variance: %.6f (expected ≈%.4f)\n", stats.variance, expected_var);
    }
    if (stats.min_val < expected_min || stats.max_val > expected_max) {
        fprintf(stderr, "  ⚠️  Range exceeded: [%.4f, %.4f] vs expected [%.2f, %.2f]\n",
                stats.min_val, stats.max_val, expected_min, expected_max);
    }
#endif
}

// Quick diagnostic macro for easy insertion
#define FWD_DIAG_BUFFER(name, buffer, elements, stream) \
    do { \
        if (FORWARD_DIAGNOSTICS_ENABLED && (buffer) && (elements) > 0) { \
            auto _stats = GRIM::Forward::computeBufferStats((buffer), (elements), (stream)); \
            _stats.print(name); \
        } \
    } while(0)

#define FWD_DIAG_BUFFER_EXPECTED(name, buffer, elements, exp_mean, exp_var, exp_min, exp_max, stream) \
    do { \
        if (FORWARD_DIAGNOSTICS_ENABLED && (buffer) && (elements) > 0) { \
            GRIM::Forward::logBufferWithExpected(name, (buffer), (elements), \
                (exp_mean), (exp_var), (exp_min), (exp_max), (stream)); \
        } \
    } while(0)

} // namespace Forward
} // namespace GRIM
