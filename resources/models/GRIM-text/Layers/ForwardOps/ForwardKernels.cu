/**
 * @file ForwardKernels.cu
 * @brief CUDA kernels for forward incremental attention and single-token ops
 *
 * Extracted from IncrementalGeneration_GPU.cu to keep forward phases modular.
 * All kernels are launched via fail-loud wrappers (no silent fallbacks).
 */

#include "ForwardKernels.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

// Helper device function: Apply RoPE rotation to a single vector element pair
__device__ __forceinline__ void applyRoPERotation(
    float& x0, float& x1,
    float cos_theta, float sin_theta
) {
    float new_x0 = x0 * cos_theta - x1 * sin_theta;
    float new_x1 = x0 * sin_theta + x1 * cos_theta;
    x0 = new_x0;
    x1 = new_x1;
}

// Kernel: Compute QK-normalized attention scores with ALiBi/RoPE/GQA support
__global__ void incrementalAttentionScoresFullKernel(
    const float* __restrict__ Q,        // [num_heads, d_head]
    const float* __restrict__ K,        // [num_kv_heads, max_kv_len, d_head] BHSD
    float* __restrict__ scores,         // [num_heads, kv_len]
    const float* __restrict__ alpha_q,  // [num_heads] (nullable)
    const float* __restrict__ alpha_k,  // [num_kv_heads] (nullable)
    const float* __restrict__ alibi_slopes, // [num_heads] (nullable)
    const float* __restrict__ rope_inv_freq, // [rotary_dim/2] (nullable)
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len,
    int query_pos,
    int rotary_dim,
    float base_scale,
    float inv_temperature,
    bool qk_norm_enabled
) {
    int q_head_idx = blockIdx.x;
    if (q_head_idx >= num_heads) return;

    int heads_per_kv_group = num_heads / num_kv_heads;
    int kv_head_idx = q_head_idx / heads_per_kv_group;

    extern __shared__ float smem[];
    float* Q_local = smem;           // [d_head]
    float* K_local = smem + d_head;  // [d_head]

    for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
        Q_local[i] = Q[q_head_idx * d_head + i];
    }
    __syncthreads();

    if (rope_inv_freq != nullptr && rotary_dim > 0 && threadIdx.x == 0) {
        for (int i = 0; i < rotary_dim / 2; ++i) {
            float theta = static_cast<float>(query_pos) * rope_inv_freq[i];
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            applyRoPERotation(Q_local[2 * i], Q_local[2 * i + 1], cos_t, sin_t);
        }
    }
    __syncthreads();

    float q_inv_norm = 1.0f;
    if (qk_norm_enabled) {
        float norm_sq = 0.0f;
        for (int i = threadIdx.x; i < d_head; i += blockDim.x) {
            norm_sq += Q_local[i] * Q_local[i];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            norm_sq += __shfl_down_sync(0xffffffff, norm_sq, offset);
        }
        norm_sq = __shfl_sync(0xffffffff, norm_sq, 0);
        q_inv_norm = rsqrtf(norm_sq + 1e-6f);
        q_inv_norm = fminf(q_inv_norm, 50.0f);
    }

    float q_scale = q_inv_norm;
    if (alpha_q != nullptr) {
        q_scale *= alpha_q[q_head_idx];
    }

    float alibi_slope = (alibi_slopes != nullptr) ? alibi_slopes[q_head_idx] : 0.0f;

    for (int k_idx = threadIdx.x; k_idx < kv_len; k_idx += blockDim.x) {
        const float* k_ptr = K + kv_head_idx * max_kv_len * d_head + k_idx * d_head;

        float score = 0.0f;
        float k_norm_sq = 0.0f;

        for (int d = 0; d < d_head; ++d) {
            float k_val = k_ptr[d];
            float q_val = Q_local[d];

            if (rope_inv_freq != nullptr && d < rotary_dim) {
                int pair_idx = d / 2;
                float theta = static_cast<float>(k_idx) * rope_inv_freq[pair_idx];
                float cos_t = cosf(theta);
                float sin_t = sinf(theta);

                if (d % 2 == 0) {
                    float k_next = k_ptr[d + 1];
                    k_val = k_val * cos_t - k_next * sin_t;
                } else {
                    float k_prev = k_ptr[d - 1];
                    k_val = k_prev * sin_t + k_val * cos_t;
                }
            }

            if (qk_norm_enabled) {
                k_norm_sq += k_val * k_val;
            }
            score += q_val * k_val;
        }

        float k_inv_norm = 1.0f;
        if (qk_norm_enabled) {
            k_inv_norm = rsqrtf(k_norm_sq + 1e-6f);
            k_inv_norm = fminf(k_inv_norm, 50.0f);
        }

        float k_scale = k_inv_norm;
        if (alpha_k != nullptr) {
            k_scale *= alpha_k[kv_head_idx];
        }

        score *= q_scale * k_scale * base_scale * inv_temperature;

        if (alibi_slopes != nullptr) {
            score += alibi_slope * static_cast<float>(k_idx - query_pos);
        }

        if (k_idx > query_pos) {
            score = -1e9f;
        }

        scores[q_head_idx * kv_len + k_idx] = score;
    }
}

// Kernel: Softmax for incremental attention (single row per head)
__global__ void incrementalSoftmaxKernel(
    float* __restrict__ scores,
    int num_heads,
    int kv_len
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    float* row = scores + head_idx * kv_len;
    float max_val = -1e9f;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i]);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        row[i] *= inv_sum;
    }
}

// Kernel: Compute attention output for single query (GQA-aware)
__global__ void incrementalAttentionOutputGQAKernel(
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    int heads_per_kv_group = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv_group;

    const float* head_scores = scores + head_idx * kv_len;

    for (int d_idx = threadIdx.x; d_idx < d_head; d_idx += blockDim.x) {
        float sum = 0.0f;
        for (int v_idx = 0; v_idx < kv_len; ++v_idx) {
            float v_val = V[kv_head_idx * max_kv_len * d_head + v_idx * d_head + d_idx];
            sum += head_scores[v_idx] * v_val;
        }
        output[head_idx * d_head + d_idx] = sum;
    }
}

// Kernel: Append new K/V to cache in BHSD format (GQA-aware)
__global__ void appendKVCacheGQAKernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_kv_heads,
    int d_head,
    int max_kv_len,
    int pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_kv_heads * d_head;
    if (idx >= total_size) return;

    int head_idx = idx / d_head;
    int d_idx = idx % d_head;
    float val = new_kv[head_idx * d_head + d_idx];
    cache[head_idx * max_kv_len * d_head + pos * d_head + d_idx] = val;
}

__global__ void singleTokenRMSNormKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int d_model,
    float eps
) {
    __shared__ float s_variance;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        float val = input[i];
        local_sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(local_sum / static_cast<float>(d_model) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        output[i] = input[i] * s_variance * gamma[i];
    }
}

__global__ void singleTokenResidualKernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_model) return;
    output[idx] = a[idx] + b[idx];
}

__global__ void singleTokenGELUKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    const float c = 0.7978845608f;
    const float a = 0.044715f;
    float x3 = x * x * x;
    output[idx] = 0.5f * x * (1.0f + tanhf(c * (x + a * x3)));
}

extern "C" void launchIncrementalAttentionScoresFull(
    const float* Q,
    const float* K,
    float* scores,
    const float* alpha_q,
    const float* alpha_k,
    const float* alibi_slopes,
    const float* rope_inv_freq,
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len,
    int query_pos,
    int rotary_dim,
    float base_scale,
    float inv_temperature,
    bool qk_norm_enabled,
    cudaStream_t stream
) {
    if (!Q || !K || !scores) {
        throw std::runtime_error("launchIncrementalAttentionScoresFull: null pointer");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || d_head <= 0 || kv_len <= 0 || max_kv_len <= 0) {
        throw std::runtime_error("launchIncrementalAttentionScoresFull: invalid dimensions");
    }
    int threads = std::min(kv_len, 256);
    size_t smem = 2 * static_cast<size_t>(d_head) * sizeof(float);
    incrementalAttentionScoresFullKernel<<<num_heads, threads, smem, stream>>>(
        Q, K, scores, alpha_q, alpha_k, alibi_slopes, rope_inv_freq,
        num_heads, num_kv_heads, d_head, kv_len, max_kv_len, query_pos,
        rotary_dim, base_scale, inv_temperature, qk_norm_enabled);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchIncrementalAttentionScoresFull kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchIncrementalSoftmax(
    float* scores,
    int num_heads,
    int kv_len,
    cudaStream_t stream
) {
    if (!scores) {
        throw std::runtime_error("launchIncrementalSoftmax: scores is NULL");
    }
    if (num_heads <= 0 || kv_len <= 0) {
        throw std::runtime_error("launchIncrementalSoftmax: invalid dimensions");
    }
    incrementalSoftmaxKernel<<<num_heads, 32, 0, stream>>>(scores, num_heads, kv_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchIncrementalSoftmax kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchIncrementalAttentionOutputGQA(
    const float* scores,
    const float* V,
    float* output,
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len,
    cudaStream_t stream
) {
    if (!scores || !V || !output) {
        throw std::runtime_error("launchIncrementalAttentionOutputGQA: null pointer");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || d_head <= 0 || kv_len <= 0 || max_kv_len <= 0) {
        throw std::runtime_error("launchIncrementalAttentionOutputGQA: invalid dimensions");
    }
    int threads = std::min(d_head, 256);
    incrementalAttentionOutputGQAKernel<<<num_heads, threads, 0, stream>>>(
        scores, V, output, num_heads, num_kv_heads, d_head, kv_len, max_kv_len);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchIncrementalAttentionOutputGQA kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchAppendKVCacheGQA(
    const float* new_kv,
    float* cache,
    int num_kv_heads,
    int d_head,
    int max_kv_len,
    int pos,
    cudaStream_t stream
) {
    if (!new_kv || !cache) {
        throw std::runtime_error("launchAppendKVCacheGQA: null pointer");
    }
    if (num_kv_heads <= 0 || d_head <= 0 || max_kv_len <= 0 || pos < 0) {
        throw std::runtime_error("launchAppendKVCacheGQA: invalid dimensions");
    }
    const int total_size = num_kv_heads * d_head;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    appendKVCacheGQAKernel<<<blocks, threads, 0, stream>>>(
        new_kv, cache, num_kv_heads, d_head, max_kv_len, pos);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchAppendKVCacheGQA kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchSingleTokenRMSNorm(
    const float* input,
    const float* gamma,
    float* output,
    int d_model,
    float eps,
    cudaStream_t stream
) {
    if (!input || !gamma || !output) {
        throw std::runtime_error("launchSingleTokenRMSNorm: null pointer");
    }
    if (d_model <= 0) {
        throw std::runtime_error("launchSingleTokenRMSNorm: invalid d_model");
    }
    const int threads = 256;
    singleTokenRMSNormKernel<<<1, threads, 0, stream>>>(input, gamma, output, d_model, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchSingleTokenRMSNorm kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchSingleTokenResidual(
    const float* a,
    const float* b,
    float* output,
    int d_model,
    cudaStream_t stream
) {
    if (!a || !b || !output) {
        throw std::runtime_error("launchSingleTokenResidual: null pointer");
    }
    if (d_model <= 0) {
        throw std::runtime_error("launchSingleTokenResidual: invalid d_model");
    }
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    singleTokenResidualKernel<<<blocks, threads, 0, stream>>>(a, b, output, d_model);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchSingleTokenResidual kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchSingleTokenGELU(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream
) {
    if (!input || !output) {
        throw std::runtime_error("launchSingleTokenGELU: null pointer");
    }
    if (size <= 0) {
        throw std::runtime_error("launchSingleTokenGELU: invalid size");
    }
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    singleTokenGELUKernel<<<blocks, threads, 0, stream>>>(input, output, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchSingleTokenGELU kernel error: " +
                                 std::string(cudaGetErrorString(err)));
    }
}
