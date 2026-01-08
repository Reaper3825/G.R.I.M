//======================================================//
//  grim_transformer_gpu.cu
//  CUDA kernels for transformer operations
//  
//  Features:
//  - Fused QKV projection kernels
//  - Optimized softmax with warp-level reduction
//  - Attention score computation with ALiBi bias
//  - FFN kernels with fused GELU
//  - Layer normalization kernels
//  - Kernel fusion for common patterns
//  - Tensor Core support (FP16/FP32 mixed precision)
//======================================================//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>
//======================================================//
//  CUDA Utility Functions
//======================================================//

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Warp-level reduction for max/sum
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Layer normalization kernels now live in
// Layers/LayernNorm/LayerNorm_Kernal_GPU.cu to keep this translation unit
// focused on attention/FFN kernels.

//======================================================//
//  Attention Kernels
//======================================================//

// GELU activation (used in FFN)
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797884560802865f * (x + 0.044715f * x * x * x)));
}

// Fused QKV projection kernel
// Combines token embedding -> Q, K, V linear projections
__global__ void fusedQKVProjectionKernel(
    const float* __restrict__ input,      // [batch_size, seq_len, d_model]
    const float* __restrict__ W_qkv,      // [3 * d_model, d_model]
    const float* __restrict__ b_qkv,      // [3 * d_model]
    float* __restrict__ output,            // [batch_size, seq_len, 3 * d_model]
    int batch_size,
    int seq_len,
    int d_model
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= 3 * d_model) {
        return;
    }
    
    const float* input_vec = input + (batch_idx * seq_len + seq_idx) * d_model;
    const float* weight_vec = W_qkv + out_idx * d_model;
    
    // Compute dot product
    float sum = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        sum += input_vec[i] * weight_vec[i];
    }
    sum += b_qkv[out_idx];
    
    output[(batch_idx * seq_len + seq_idx) * 3 * d_model + out_idx] = sum;
}

// Scaled dot-product attention scores
__global__ void attentionScoresKernel(
    const float* __restrict__ Q,           // [batch, heads, seq_len, d_head]
    const float* __restrict__ K,           // [batch, heads, seq_len, d_head]
    float* __restrict__ scores,            // [batch, heads, seq_len, seq_len]
    const float* __restrict__ alibi_bias,  // [heads, seq_len, seq_len] (optional)
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    float scale,
    bool use_alibi,
    bool causal_mask
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x;
    int k_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        q_idx >= seq_len || k_idx >= seq_len) {
        return;
    }
    
    // Causal masking: only attend to previous positions
    if (causal_mask && k_idx > q_idx) {
        scores[((batch_idx * num_heads + head_idx) * seq_len + q_idx) * seq_len + k_idx] = -1e9f;
        return;
    }
    
    const float* q_vec = Q + ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * d_head;
    const float* k_vec = K + ((batch_idx * num_heads + head_idx) * seq_len + k_idx) * d_head;
    
    // Compute Q · K^T
    float score = 0.0f;
    for (int i = 0; i < d_head; ++i) {
        score += q_vec[i] * k_vec[i];
    }
    score *= scale;
    
    // Add ALiBi bias if enabled
    if (use_alibi && alibi_bias != nullptr) {
        score += alibi_bias[(head_idx * seq_len + q_idx) * seq_len + k_idx];
    } else { std::cout << "ALiBi bias is null!" << std::endl; }
    
    scores[((batch_idx * num_heads + head_idx) * seq_len + q_idx) * seq_len + k_idx] = score;
}

// Optimized softmax kernel with warp-level reduction
__global__ void softmaxKernel(
    float* __restrict__ scores,  // [batch, heads, seq_len, seq_len]
    int batch_size,
    int num_heads,
    int seq_len
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || row_idx >= seq_len) {
        return;
    }
    
    float* row = scores + ((batch_idx * num_heads + head_idx) * seq_len + row_idx) * seq_len;
    
    // Find max value for numerical stability
    float max_val = -1e9f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }
    max_val = warpReduceMax(max_val);
    
    // Cross-warp reduction for max
    __shared__ float shared_max[32];  // One per warp
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;
    
    if (lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (warp_id == 0) {
        max_val = (threadIdx.x < num_warps) ? shared_max[threadIdx.x] : -1e9f;
        max_val = warpReduceMax(max_val);
        if (threadIdx.x == 0) {
            shared_max[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float exp_val = expf(row[i] - max_val);
        row[i] = exp_val;
        sum += exp_val;
    }
    sum = warpReduceSum(sum);
    
    // Cross-warp reduction for sum
    __shared__ float shared_sum[32];  // One per warp
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    // First warp reduces across all warps
    if (warp_id == 0) {
        sum = (threadIdx.x < num_warps) ? shared_sum[threadIdx.x] : 0.0f;
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0) {
            shared_sum[0] = sum;
        }
    }
    __syncthreads();
    sum = shared_sum[0];
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

// Attention output: scores * V
__global__ void attentionOutputKernel(
    const float* __restrict__ scores,  // [batch, heads, seq_len, seq_len]
    const float* __restrict__ V,       // [batch, heads, seq_len, d_head]
    float* __restrict__ output,        // [batch, heads, seq_len, d_head]
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int d_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        seq_idx >= seq_len || d_idx >= d_head) {
        return;
    }
    
    const float* score_row = scores + ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * seq_len;
    float sum = 0.0f;
    
    for (int i = 0; i < seq_len; ++i) {
        const float* v_vec = V + ((batch_idx * num_heads + head_idx) * seq_len + i) * d_head;
        sum += score_row[i] * v_vec[d_idx];
    }
    
    output[((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * d_head + d_idx] = sum;
}

//======================================================//
//  Feed-Forward Network Kernels
//======================================================//

// Fused FFN first layer with GELU (W1 * x + b1, then GELU)
__global__ void ffnLayer1GeluKernel(
    const float* __restrict__ input,   // [batch_size, seq_len, d_model]
    const float* __restrict__ W1,      // [d_ff, d_model]
    const float* __restrict__ b1,      // [d_ff]
    float* __restrict__ output,        // [batch_size, seq_len, d_ff]
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= d_ff) {
        return;
    }
    
    const float* input_vec = input + (batch_idx * seq_len + seq_idx) * d_model;
    const float* weight_vec = W1 + out_idx * d_model;
    
    // Compute W1 * x + b1
    float sum = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        sum += input_vec[i] * weight_vec[i];
    }
    sum += b1[out_idx];
    
    // Apply GELU activation
    sum = gelu(sum);
    
    output[(batch_idx * seq_len + seq_idx) * d_ff + out_idx] = sum;
}

// FFN second layer (W2 * x + b2)
__global__ void ffnLayer2Kernel(
    const float* __restrict__ input,   // [batch_size, seq_len, d_ff]
    const float* __restrict__ W2,      // [d_model, d_ff]
    const float* __restrict__ b2,      // [d_model]
    float* __restrict__ output,        // [batch_size, seq_len, d_model]
    int batch_size,
    int seq_len,
    int d_ff,
    int d_model
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || out_idx >= d_model) {
        return;
    }
    
    const float* input_vec = input + (batch_idx * seq_len + seq_idx) * d_ff;
    const float* weight_vec = W2 + out_idx * d_ff;
    
    // Compute W2 * x + b2
    float sum = 0.0f;
    for (int i = 0; i < d_ff; ++i) {
        sum += input_vec[i] * weight_vec[i];
    }
    sum += b2[out_idx];
    
    output[(batch_idx * seq_len + seq_idx) * d_model + out_idx] = sum;
}

//======================================================//
//  Residual Connection Kernel
//======================================================//

__global__ void residualAddKernel(
    const float* input,
    const float* residual,
    float* output,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx] + residual[idx];
    }
}

// Scaled residual addition to prevent vanishing/exploding through deep networks
// output = input + scale * residual
__global__ void residualAddScaledKernel(
    const float* input,
    const float* residual,
    float* output,
    float scale,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx] + scale * residual[idx];
    }
}

//======================================================//
//  Launch Wrappers
//======================================================//

extern "C" void launchFusedQKVProjection(
    const float* input,
    const float* W_qkv,
    const float* b_qkv,
    float* output,
    int batch_size,
    int seq_len,
    int d_model,
    cudaStream_t stream
) {
    int output_dim = 3 * d_model;
    dim3 grid((output_dim + 255) / 256, seq_len, batch_size);
    dim3 block(256);
    
    fusedQKVProjectionKernel<<<grid, block, 0, stream>>>(
        input, W_qkv, b_qkv, output, batch_size, seq_len, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchAttentionScores(
    const float* Q,
    const float* K,
    float* scores,
    const float* alibi_bias,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    float scale,
    bool use_alibi,
    bool causal_mask,
    cudaStream_t stream
) {
    // Validate dimensions to avoid "invalid argument" CUDA errors
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || d_head <= 0) {
        fprintf(stderr, "launchAttentionScores: invalid dimensions batch=%d heads=%d seq=%d d_head=%d\n",
                batch_size, num_heads, seq_len, d_head);
        return;
    }
    
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(min(seq_len, 1024));
    
    attentionScoresKernel<<<grid, block, 0, stream>>>(
        Q, K, scores, alibi_bias, batch_size, num_heads, seq_len, d_head,
        scale, use_alibi, causal_mask
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchSoftmax(
    float* scores,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(min(1024, ((seq_len + 31) / 32) * 32));
    
    softmaxKernel<<<grid, block, 0, stream>>>(
        scores, batch_size, num_heads, seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchAttentionOutput(
    const float* scores,
    const float* V,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(min(d_head, 1024));
    
    attentionOutputKernel<<<grid, block, 0, stream>>>(
        scores, V, output, batch_size, num_heads, seq_len, d_head
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchFFNLayer1Gelu(
    const float* input,
    const float* W1,
    const float* b1,
    float* output,
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff,
    cudaStream_t stream
) {
    dim3 grid((d_ff + 255) / 256, seq_len, batch_size);
    dim3 block(256);
    
    ffnLayer1GeluKernel<<<grid, block, 0, stream>>>(
        input, W1, b1, output, batch_size, seq_len, d_model, d_ff
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchFFNLayer2(
    const float* input,
    const float* W2,
    const float* b2,
    float* output,
    int batch_size,
    int seq_len,
    int d_ff,
    int d_model,
    cudaStream_t stream
) {
    dim3 grid((d_model + 255) / 256, seq_len, batch_size);
    dim3 block(256);
    
    ffnLayer2Kernel<<<grid, block, 0, stream>>>(
        input, W2, b2, output, batch_size, seq_len, d_ff, d_model
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchResidualAdd(
    const float* input,
    const float* residual,
    float* output,
    int total_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    residualAddKernel<<<blocks, threads, 0, stream>>>(
        input, residual, output, total_size
    );
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void launchResidualAddScaled(
    const float* input,
    const float* residual,
    float* output,
    float scale,
    int total_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    residualAddScaledKernel<<<blocks, threads, 0, stream>>>(
        input, residual, output, scale, total_size
    );
    CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  BACKWARD PASS KERNELS
//======================================================//

// Layer norm backward kernels reside in Layers/LayernNorm/LayerNorm_Kernal_GPU.cu.
// GELU backward now uses Shared/Activations/GELU/GELU.cu (GRIM::launchGeluBackward)

//======================================================//
//  GELU Forward Kernel (separate from linear layer)
//======================================================//

__global__ void geluForwardKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    float x = input[idx];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x_cubed = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x_cubed);
    output[idx] = 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" void launchGelu(
    const float* input,
    float* output,
    int total_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    geluForwardKernel<<<blocks, threads, 0, stream>>>(
        input, output, total_size
    );
    CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  Add Bias Kernel (broadcast bias across batch/sequence)
//======================================================//

__global__ void addBiasKernel(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int total_tokens = batch_size * seq_len;
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= hidden_dim) return;
    
    int idx = token_idx * hidden_dim + dim_idx;
    data[idx] += bias[dim_idx];
}

extern "C" void launchAddBias(
    float* data,
    const float* bias,
    int batch_size,
    int seq_len,
    int hidden_dim,
    cudaStream_t stream
) {
    int total_tokens = batch_size * seq_len;
    dim3 grid(total_tokens, (hidden_dim + 255) / 256);
    dim3 block(min(256, hidden_dim));
    
    addBiasKernel<<<grid, block, 0, stream>>>(
        data, bias, batch_size, seq_len, hidden_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// Legacy launchGeluBackward removed - use GRIM::launchGeluBackward from Shared/Activations/GELU/GELU.hpp

// NOTE: launchBiasSumGradient has been removed from this file
// Use the production implementation in BackwardKernels.cu

//======================================================//
//  FFN Layer 2 Backward (Linear layer)
//======================================================//

// This is a simple matrix transpose for weight gradients
// grad_W = hidden^T @ grad_output
// grad_bias = sum(grad_output)
// grad_hidden = grad_output @ W^T

__global__ void ffnLayer2BackwardBiasKernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int batch_size,
    int seq_len,
    int output_dim
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= output_dim) return;
    
    float sum = 0.0f;
    for (int b = 0; b < batch_size * seq_len; b++) {
        sum += grad_output[b * output_dim + out_idx];
    }
    
    atomicAdd(&grad_bias[out_idx], sum);
}

extern "C" void launchFFNLayer2BackwardBias(
    const float* grad_output,
    float* grad_bias,
    int batch_size,
    int seq_len,
    int output_dim,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (output_dim + threads - 1) / threads;
    
    ffnLayer2BackwardBiasKernel<<<blocks, threads, 0, stream>>>(
        grad_output, grad_bias, batch_size, seq_len, output_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// NOTE: Adam optimizer kernel and launch function moved to grim_training_kernels.cu
// to avoid duplication with training-specific kernels

//======================================================//
//  Attention Backward Kernels
//======================================================//

// Backward through attention output: grad(scores @ V)
// Given grad_attn_out, compute grad_scores and grad_V
__global__ void attentionOutputBackwardKernel(
    const float* grad_attn_out,  // [batch, heads, seq, d_head]
    const float* scores,          // [batch, heads, seq, seq]
    const float* V,               // [batch, heads, seq, d_head]
    float* grad_scores,           // [batch, heads, seq, seq]
    float* grad_V,                // [batch, heads, seq, d_head]
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * seq_len;
    
    if (idx < total_elements) {
        int b = idx / (num_heads * seq_len * seq_len);
        int remainder = idx % (num_heads * seq_len * seq_len);
        int h = remainder / (seq_len * seq_len);
        remainder = remainder % (seq_len * seq_len);
        int i = remainder / seq_len;  // query position
        int j = remainder % seq_len;   // key position
        
        // grad_scores[b,h,i,j] = sum_d(grad_attn_out[b,h,i,d] * V[b,h,j,d])
        float grad_score = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            int grad_out_idx = ((b * num_heads + h) * seq_len + i) * d_head + d;
            int v_idx = ((b * num_heads + h) * seq_len + j) * d_head + d;
            grad_score += grad_attn_out[grad_out_idx] * V[v_idx];
        }
        grad_scores[idx] = grad_score;
        
        // grad_V[b,h,j,d] = sum_i(scores[b,h,i,j] * grad_attn_out[b,h,i,d])
        // This needs atomic add since multiple threads write to same grad_V
        if (i == 0) {  // Only one thread per (b,h,j) computes grad_V
            for (int d = 0; d < d_head; ++d) {
                float grad_v = 0.0f;
                for (int ii = 0; ii < seq_len; ++ii) {
                    int score_idx = ((b * num_heads + h) * seq_len + ii) * seq_len + j;
                    int grad_out_idx = ((b * num_heads + h) * seq_len + ii) * d_head + d;
                    grad_v += scores[score_idx] * grad_attn_out[grad_out_idx];
                }
                int v_idx = ((b * num_heads + h) * seq_len + j) * d_head + d;
                atomicAdd(&grad_V[v_idx], grad_v);
            }
        }
    }
}

extern "C" void launchAttentionOutputBackward(
    const float* grad_attn_out,
    const float* scores,
    const float* V,
    float* grad_scores,
    float* grad_V,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    attentionOutputBackwardKernel<<<blocks, threads, 0, stream>>>(
        grad_attn_out, scores, V, grad_scores, grad_V,
        batch_size, num_heads, seq_len, d_head
    );
    CUDA_CHECK(cudaGetLastError());
}

// Backward through softmax
// grad_pre_softmax[i] = scores[i] * (grad_scores[i] - sum_j(grad_scores[j] * scores[j]))
__global__ void softmaxBackwardKernel(
    const float* grad_scores,      // [batch, heads, seq, seq]
    const float* scores,            // [batch, heads, seq, seq] (post-softmax)
    float* grad_pre_softmax,        // [batch, heads, seq, seq]
    int batch_size,
    int num_heads,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = batch_size * num_heads * seq_len;
    
    if (idx < total_rows) {
        int b = idx / (num_heads * seq_len);
        int remainder = idx % (num_heads * seq_len);
        int h = remainder / seq_len;
        int i = remainder % seq_len;
        
        int base_idx = ((b * num_heads + h) * seq_len + i) * seq_len;
        
        // Compute sum_j(grad_scores[i,j] * scores[i,j])
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            sum += grad_scores[base_idx + j] * scores[base_idx + j];
        }
        
        // grad_pre_softmax[i,j] = scores[i,j] * (grad_scores[i,j] - sum)
        for (int j = 0; j < seq_len; ++j) {
            grad_pre_softmax[base_idx + j] = scores[base_idx + j] * (grad_scores[base_idx + j] - sum);
        }
    }
}

extern "C" void launchSoftmaxBackward(
    const float* grad_scores,
    const float* scores,
    float* grad_pre_softmax,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    softmaxBackwardKernel<<<blocks, threads, 0, stream>>>(
        grad_scores, scores, grad_pre_softmax,
        batch_size, num_heads, seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}

// Backward through attention scores: scores = Q @ K^T * scale
// grad_Q[b,h,i,d] = sum_j(grad_scores[b,h,i,j] * K[b,h,j,d]) * scale
// grad_K[b,h,j,d] = sum_i(grad_scores[b,h,i,j] * Q[b,h,i,d]) * scale
__global__ void attentionScoresBackwardKernel(
    const float* grad_scores,  // [batch, heads, seq, seq]
    const float* Q,             // [batch, heads, seq, d_head]
    const float* K,             // [batch, heads, seq, d_head]
    float* grad_Q,              // [batch, heads, seq, d_head]
    float* grad_K,              // [batch, heads, seq, d_head]
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_qk = batch_size * num_heads * seq_len * d_head;
    
    if (idx < total_qk) {
        int b = idx / (num_heads * seq_len * d_head);
        int remainder = idx % (num_heads * seq_len * d_head);
        int h = remainder / (seq_len * d_head);
        remainder = remainder % (seq_len * d_head);
        int i = remainder / d_head;
        int d = remainder % d_head;
        
        // grad_Q[b,h,i,d] = sum_j(grad_scores[b,h,i,j] * K[b,h,j,d]) * scale
        float grad_q = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            int score_idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
            int k_idx = ((b * num_heads + h) * seq_len + j) * d_head + d;
            grad_q += grad_scores[score_idx] * K[k_idx];
        }
        grad_Q[idx] = grad_q * scale;
        
        // grad_K[b,h,j,d] = sum_i(grad_scores[b,h,i,j] * Q[b,h,i,d]) * scale
        // Note: idx represents position (b,h,i,d) for Q, and (b,h,j,d) for K
        // So we reuse i as j for K computation
        float grad_k = 0.0f;
        for (int ii = 0; ii < seq_len; ++ii) {
            int score_idx = ((b * num_heads + h) * seq_len + ii) * seq_len + i;  // i is the key position
            int q_idx = ((b * num_heads + h) * seq_len + ii) * d_head + d;
            grad_k += grad_scores[score_idx] * Q[q_idx];
        }
        grad_K[idx] = grad_k * scale;
    }
}

extern "C" void launchAttentionScoresBackward(
    const float* grad_scores,
    const float* Q,
    const float* K,
    float* grad_Q,
    float* grad_K,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    float scale,
    cudaStream_t stream
) {
    int total = batch_size * num_heads * seq_len * d_head;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    attentionScoresBackwardKernel<<<blocks, threads, 0, stream>>>(
        grad_scores, Q, K, grad_Q, grad_K,
        batch_size, num_heads, seq_len, d_head, scale
    );
    CUDA_CHECK(cudaGetLastError());
}
