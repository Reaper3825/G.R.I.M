#include "ForwardDiagnostics.cuh"
#include <algorithm>
#include <vector>

namespace GRIM {
namespace Forward {

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
BufferStats computeBufferStats(
    const float* d_buffer,
    size_t num_elements,
    cudaStream_t stream
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
        stats.mean = h_sum / static_cast<float>(valid_count);
        stats.variance = (h_sq_sum / static_cast<float>(valid_count)) - (stats.mean * stats.mean);
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

// ═══════════════════════════════════════════════════════════════════════════
// TOKEN 277 ALIGNMENT KERNEL (Issue #37)
// Computes dot product and cosine similarity between each hidden state and W[277]
// ═══════════════════════════════════════════════════════════════════════════

__global__ void computeToken277AlignmentKernel(
    const float* __restrict__ hidden,    // [total_tokens, d_model] row-major
    const float* __restrict__ w277,      // [d_model]
    int total_tokens,
    int d_model,
    float* __restrict__ out_dot_sum,     // Sum of dot products
    float* __restrict__ out_dot_max,     // Max dot product
    float* __restrict__ out_cos_sum,     // Sum of cosine similarities
    float* __restrict__ out_cos_max,     // Max cosine similarity
    float* __restrict__ out_hnorm_sum,   // Sum of hidden norms
    float w277_norm                      // Pre-computed W[277] norm
) {
    __shared__ float s_dot_sum[256];
    __shared__ float s_dot_max[256];
    __shared__ float s_cos_sum[256];
    __shared__ float s_cos_max[256];
    __shared__ float s_hnorm_sum[256];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    float local_dot_sum = 0.0f;
    float local_dot_max = -1e30f;
    float local_cos_sum = 0.0f;
    float local_cos_max = -1.0f;
    float local_hnorm_sum = 0.0f;
    
    // Each thread processes one or more tokens
    for (int token = gid; token < total_tokens; token += stride) {
        const float* h = hidden + static_cast<size_t>(token) * d_model;
        
        // Compute dot product and hidden norm
        float dot = 0.0f;
        float h_sq_sum = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            dot += h[i] * w277[i];
            h_sq_sum += h[i] * h[i];
        }
        float h_norm = sqrtf(h_sq_sum + 1e-8f);
        
        // Compute cosine similarity
        float cos_sim = dot / (h_norm * w277_norm + 1e-8f);
        
        local_dot_sum += dot;
        local_dot_max = fmaxf(local_dot_max, dot);
        local_cos_sum += cos_sim;
        local_cos_max = fmaxf(local_cos_max, cos_sim);
        local_hnorm_sum += h_norm;
    }
    
    // Store to shared memory
    s_dot_sum[tid] = local_dot_sum;
    s_dot_max[tid] = local_dot_max;
    s_cos_sum[tid] = local_cos_sum;
    s_cos_max[tid] = local_cos_max;
    s_hnorm_sum[tid] = local_hnorm_sum;
    __syncthreads();
    
    // Reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot_sum[tid] += s_dot_sum[tid + s];
            s_dot_max[tid] = fmaxf(s_dot_max[tid], s_dot_max[tid + s]);
            s_cos_sum[tid] += s_cos_sum[tid + s];
            s_cos_max[tid] = fmaxf(s_cos_max[tid], s_cos_max[tid + s]);
            s_hnorm_sum[tid] += s_hnorm_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result atomically
    if (tid == 0) {
        atomicAdd(out_dot_sum, s_dot_sum[0]);
        atomicAdd(out_cos_sum, s_cos_sum[0]);
        atomicAdd(out_hnorm_sum, s_hnorm_sum[0]);
        
        // Atomic max for floats
        float old_max = *out_dot_max;
        while (s_dot_max[0] > old_max) {
            old_max = __int_as_float(atomicCAS(
                (int*)out_dot_max, 
                __float_as_int(old_max), 
                __float_as_int(s_dot_max[0])
            ));
        }
        old_max = *out_cos_max;
        while (s_cos_max[0] > old_max) {
            old_max = __int_as_float(atomicCAS(
                (int*)out_cos_max, 
                __float_as_int(old_max), 
                __float_as_int(s_cos_max[0])
            ));
        }
    }
}

Token277AlignmentStats computeToken277Alignment(
    const float* d_hidden,
    const float* d_w277,
    int total_tokens,
    int d_model,
    cudaStream_t stream
) {
    // DISABLED - set to true to enable [Token277Align] logs
    constexpr bool kEnableToken277AlignDiag = false;
    
    Token277AlignmentStats stats = {};
    stats.num_tokens = total_tokens;
    
    if constexpr (!kEnableToken277AlignDiag) {
        return stats;
    }
    
    fprintf(stderr, "[Token277Align] ENTRY: d_hidden=%p d_w277=%p tokens=%d d_model=%d\n",
            (void*)d_hidden, (void*)d_w277, total_tokens, d_model);
    
    if (!d_hidden || !d_w277 || total_tokens <= 0 || d_model <= 0) {
        fprintf(stderr, "[Token277Align] EARLY_EXIT: null or zero params\n");
        return stats;
    }
    
    // First compute W[277] norm on host (one-time, small)
    std::vector<float> h_w277(d_model);
    cudaMemcpy(h_w277.data(), d_w277, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    float w277_sq_sum = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        w277_sq_sum += h_w277[i] * h_w277[i];
    }
    stats.w277_norm = sqrtf(w277_sq_sum + 1e-8f);
    
    // Allocate device memory for reduction
    float* d_dot_sum;
    float* d_dot_max;
    float* d_cos_sum;
    float* d_cos_max;
    float* d_hnorm_sum;
    
    cudaMalloc(&d_dot_sum, sizeof(float));
    cudaMalloc(&d_dot_max, sizeof(float));
    cudaMalloc(&d_cos_sum, sizeof(float));
    cudaMalloc(&d_cos_max, sizeof(float));
    cudaMalloc(&d_hnorm_sum, sizeof(float));
    
    // Initialize
    float zero = 0.0f;
    float neg_inf = -1e30f;
    float neg_one = -1.0f;
    cudaMemcpyAsync(d_dot_sum, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dot_max, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cos_sum, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cos_max, &neg_one, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_hnorm_sum, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    const int block_size = 256;
    const int num_blocks = std::min((total_tokens + block_size - 1) / block_size, 1024);
    computeToken277AlignmentKernel<<<num_blocks, block_size, 0, stream>>>(
        d_hidden, d_w277, total_tokens, d_model,
        d_dot_sum, d_dot_max, d_cos_sum, d_cos_max, d_hnorm_sum,
        stats.w277_norm
    );
    
    // Copy results back
    float h_dot_sum, h_dot_max, h_cos_sum, h_cos_max, h_hnorm_sum;
    cudaMemcpyAsync(&h_dot_sum, d_dot_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_dot_max, d_dot_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_cos_sum, d_cos_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_cos_max, d_cos_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_hnorm_sum, d_hnorm_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Compute means
    stats.dot_product_mean = h_dot_sum / total_tokens;
    stats.dot_product_max = h_dot_max;
    stats.cosine_sim_mean = h_cos_sum / total_tokens;
    stats.cosine_sim_max = h_cos_max;
    stats.hidden_norm_mean = h_hnorm_sum / total_tokens;
    
    // Cleanup
    cudaFree(d_dot_sum);
    cudaFree(d_dot_max);
    cudaFree(d_cos_sum);
    cudaFree(d_cos_max);
    cudaFree(d_hnorm_sum);
    
    return stats;
}

} // namespace Forward
} // namespace GRIM
