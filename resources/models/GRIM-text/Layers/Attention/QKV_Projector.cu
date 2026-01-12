#include "QKV_Projector.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../FlashAttention/AttentionDiagnostics.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace {

// 1D grid bias kernel: launch-safe for arbitrary total_tokens (no grid.y limit)
// Uses optimized indexing: token = idx/channels, channel = idx - token*channels
// Avoids expensive modulo operation while supporting unlimited tokens
__global__ void addBiasKernel(float* data,
                              const float* bias,
                              int total_tokens,
                              int channels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_tokens * channels;
    if (idx >= total) {
        return;
    }
    
    // Optimized decode: avoids modulo, same speed as 2D grid
    const int token = idx / channels;
    const int channel = idx - token * channels;
    
    data[idx] += bias[channel];
}

// Reshape Q/K/V from [tokens, src_stride] to [batch, heads, seq, head_dim]
// Optimized for coalesced memory access
// src_stride: actual stride of source tensor (may differ from num_heads * head_dim)
__global__ void reshapeQKVToBHSDKernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int batch_size,
                                       int seq_len,
                                       int num_heads,
                                       int head_dim,
                                       int src_stride) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;
    if (idx >= total) return;

    // Decode output [batch, heads, seq, head_dim] index
    const int d = idx % head_dim;
    int tmp = idx / head_dim;
    const int s = tmp % seq_len;
    tmp /= seq_len;
    const int h = tmp % num_heads;
    const int b = tmp / num_heads;

    // Calculate source position in [batch*seq, src_stride] format
    // Features are organized as [head0_features, head1_features, ..., headN_features]
    const int token_idx = b * seq_len + s;
    const int src_idx = token_idx * src_stride + h * head_dim + d;
    
    // Coalesced write to output
    dst[idx] = src[src_idx];
}

// Reshape attention output from [batch, heads, seq, head_dim] back to [tokens, d_model]
// Inverse of reshapeQKVToBHSDKernel
__global__ void reshapeFromBHSDKernel(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int batch_size,
                                     int seq_len,
                                     int num_heads,
                                     int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int d_model = num_heads * head_dim;
    const int total = batch_size * seq_len * d_model;
    if (idx >= total) return;

    // Decode output [tokens, d_model] index
    const int token_idx = idx / d_model;
    const int feature_idx = idx % d_model;
    const int h = feature_idx / head_dim;
    const int d = feature_idx % head_dim;

    // Decode batch and sequence indices from token_idx
    const int b = token_idx / seq_len;
    const int s = token_idx % seq_len;

    // Calculate source position in [batch, heads, seq, head_dim] format
    const int src_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
    
    // Coalesced write to output
    dst[idx] = src[src_idx];
}

inline void addBias(float* data,
                    const float* bias,
                    int total_tokens,
                    int channels,
                    cudaStream_t stream) {
    if (!data || !bias || total_tokens <= 0 || channels <= 0) {
        return;
    }
    // 1D grid: no grid.y limit, optimized indexing (no modulo)
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int total = total_tokens * channels;
    const int grid = (total + kBlockSize - 1) / kBlockSize;
    addBiasKernel<<<grid, kBlockSize, 0, stream>>>(data, bias, total_tokens, channels);
    
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[QKV_Projector] addBias kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace

size_t getQkvProjectionWorkspaceSize(const QKVProjectionConfig& config) {
    const int total_tokens = config.batch_size * config.seq_len;
    if (total_tokens <= 0 || config.d_model <= 0) {
        return 0;
    }
    return static_cast<size_t>(total_tokens) * 3 * config.d_model * sizeof(float);
}

void launchQkvProjection(const float* input,
                         const QKVProjectionWeights& weights,
                         float* q_out,
                         float* k_out,
                         float* v_out,
                         float* workspace,
                         const QKVProjectionConfig& config) {
    // Rule 20: Fail loud on null pointers
    if (!input) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: input is NULL");
    }
    if (!weights.W_qkv) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: weights.W_qkv is NULL");
    }
    if (!q_out) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: q_out is NULL");
    }
    if (!k_out) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: k_out is NULL");
    }
    if (!v_out) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: v_out is NULL");
    }

    const int total_tokens = config.batch_size * config.seq_len;
    const int d_model = config.d_model;
    
    // Rule 20: Fail loud on invalid dimensions
    if (total_tokens <= 0 || d_model <= 0 || config.num_heads <= 0) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: Invalid dimensions "
                "tokens=" + std::to_string(total_tokens) + " d_model=" + std::to_string(d_model) +
                " num_heads=" + std::to_string(config.num_heads));
    }
    
    // Rule 20: Fail loud on head divisibility
    if (d_model % config.num_heads != 0) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: d_model (" +
                std::to_string(d_model) + ") must be divisible by num_heads (" +
                std::to_string(config.num_heads) + ")");
    }
    
    // Rule 20: Fail loud on missing stream
    if (!config.stream) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: config.stream is NULL - caller MUST provide valid stream");
    }

    const size_t workspace_bytes = getQkvProjectionWorkspaceSize(config);
    // Rule 20: Require workspace - no silent allocation fallback
    if (!workspace) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: workspace is NULL - caller MUST provide pre-allocated workspace");
    }
    float* fused_qkv = workspace;

    cublasHandle_t handle = config.handle;
    if (!handle) {
        throw std::runtime_error("[QKV_Projector] launchQkvProjection: config.handle is NULL - MUST pass training_state.cublas_handle per Rule 22");
    }
    // CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
    cublasSetStream(handle, config.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // =========================================================================
    // ROW-MAJOR WEIGHT CONVENTION (CRITICAL - DO NOT REFACTOR WITHOUT CARE)
    // =========================================================================
    // All weight matrices in GRIM-text are stored ROW-MAJOR. This affects how
    // we call cuBLAS, which assumes column-major by default.
    //
    // Layout:
    //   input:     [tokens, d_model]   row-major, stride=d_model
    //   W_qkv:     [3*d_model, d_model] row-major, stride=d_model
    //   fused_qkv: [tokens, 3*d_model]  row-major, stride=3*d_model
    //
    // Mathematical operation: fused_qkv = input @ W_qkv^T
    //   Shape: [tokens, 3*d_model] = [tokens, d_model] @ [d_model, 3*d_model]
    //
    // cuBLAS row-major trick: To compute C = A @ B^T with row-major matrices,
    // we call sgemm with swapped arguments and OP_T on B:
    //   sgemm(OP_T, OP_N, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
    //
    // Parameters: M=tokens, N=3*d_model, K=d_model
    //   A = input [tokens, d_model], lda = d_model
    //   B = W_qkv [3*d_model, d_model], ldb = d_model (transposed via OP_T)
    //   C = fused_qkv [tokens, 3*d_model], ldc = 3*d_model
    //
    // WARNING: This pattern ONLY works with row-major weights. If you switch
    // to column-major or cublasLt, this code must be rewritten.
    // =========================================================================
    cublasStatus_t status = cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                3 * d_model,              // N: cols of result
                total_tokens,             // M: rows of result
                d_model,                  // K: shared dimension
                &alpha,
                weights.W_qkv,            // B: W_qkv [3*d_model, d_model], will be transposed
                d_model,                  // ldb = d_model (row stride of W_qkv)
                input,                    // A: input [tokens, d_model]
                d_model,                  // lda = d_model (row stride of input)
                &beta,
                fused_qkv,                // C: fused_qkv [tokens, 3*d_model]
                3 * d_model);             // ldc = 3*d_model (row stride of output)
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[QKV_Projector] cublasSgemm failed: status=" + std::to_string(status) +
                " m=" + std::to_string(3 * d_model) + " n=" + std::to_string(total_tokens) +
                " k=" + std::to_string(d_model));
    }

    addBias(fused_qkv, weights.b_qkv, total_tokens, 3 * d_model, config.stream);

    // Row-major [tokens, 3*d_model]: each row has [Q_features | K_features | V_features]
    // Split by strided 2D copy - faster than a custom kernel for this pattern
    // 
    // Memory layout per token row: [q0..q_{d-1} | k0..k_{d-1} | v0..v_{d-1}]
    // We extract each component with cudaMemcpy2DAsync using stride=3*d_model
    {
        // Split kernel: extract Q, K, V from interleaved row-major format
        auto split_qkv = [&](float* dst, int offset, const char* name) {
            // For each (token, feature) in [tokens, d_model]:
            //   dst[token * d_model + feature] = fused_qkv[token * 3*d_model + offset + feature]
            cudaError_t err = cudaMemcpy2DAsync(dst, d_model * sizeof(float),
                             fused_qkv + offset, 3 * d_model * sizeof(float),
                             d_model * sizeof(float), total_tokens,
                             cudaMemcpyDeviceToDevice, config.stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("[QKV_Projector] cudaMemcpy2DAsync for ") + name +
                        " failed: " + cudaGetErrorString(err));
            }
        };
        
        split_qkv(q_out, 0, "Q");
        split_qkv(k_out, d_model, "K");
        split_qkv(v_out, 2 * d_model, "V");
    }

    // Rule 22: Do NOT destroy handle - we don't own it!
}

// GQA-specific projection using separate W_q, W_k, W_v
void launchGQAProjection(const float* input,
                         const QKVProjectionWeights& weights,
                         float* q_out,
                         float* k_out,
                         float* v_out,
                         const QKVProjectionConfig& config) {
    // Rule 20: Fail loud on null pointers
    if (!input) {
        throw std::runtime_error("[GQAProjection] input is NULL");
    }
    if (!q_out) {
        throw std::runtime_error("[GQAProjection] q_out is NULL");
    }
    if (!k_out) {
        throw std::runtime_error("[GQAProjection] k_out is NULL");
    }
    if (!v_out) {
        throw std::runtime_error("[GQAProjection] v_out is NULL");
    }
    
    const int total_tokens = config.batch_size * config.seq_len;
    const int d_model = config.d_model;
    
    // Rule 20: Fail loud on invalid dimensions
    if (total_tokens <= 0 || d_model <= 0 || config.num_heads <= 0) {
        throw std::runtime_error("[GQAProjection] Invalid dimensions tokens=" +
                std::to_string(total_tokens) + " d_model=" + std::to_string(d_model) +
                " num_heads=" + std::to_string(config.num_heads));
    }
    
    if (d_model % config.num_heads != 0) {
        throw std::runtime_error("[GQAProjection] d_model (" + std::to_string(d_model) +
                ") must be divisible by num_heads (" + std::to_string(config.num_heads) + ")");
    }
    
    const int num_kv_heads = config.getNumKVHeads();
    if (num_kv_heads > config.num_heads) {
        throw std::runtime_error("[GQAProjection] num_kv_heads (" + std::to_string(num_kv_heads) +
                ") cannot exceed num_heads (" + std::to_string(config.num_heads) + ")");
    }
    if (config.num_heads % num_kv_heads != 0) {
        throw std::runtime_error("[GQAProjection] num_heads (" + std::to_string(config.num_heads) +
                ") must be divisible by num_kv_heads (" + std::to_string(num_kv_heads) + ")");
    }
    
    // Rule 20: Require explicit GQA weights - NO FALLBACKS
    if (!weights.W_q) {
        throw std::runtime_error("[GQAProjection] weights.W_q is NULL - GQA mode requires separate Q weight matrix");
    }
    if (!weights.W_k) {
        throw std::runtime_error("[GQAProjection] weights.W_k is NULL - GQA mode requires separate K weight matrix");
    }
    if (!weights.W_v) {
        throw std::runtime_error("[GQAProjection] weights.W_v is NULL - GQA mode requires separate V weight matrix");
    }
    
    const float* W_q = weights.W_q;
    const float* W_k = weights.W_k;
    const float* W_v = weights.W_v;
    const float* b_q = weights.b_q;  // Biases are optional
    const float* b_k = weights.b_k;
    const float* b_v = weights.b_v;
    
    const int head_dim = d_model / config.num_heads;
    const int kv_dim = num_kv_heads * head_dim;
    
    cublasHandle_t handle = config.handle;
    if (!handle) {
        throw std::runtime_error("[GQAProjection] config.handle is NULL - MUST pass training_state.cublas_handle per Rule 22");
    }
    // CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
    cublasSetStream(handle, config.stream);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;
    
    // Q projection: [tokens, d_model] @ W_q^T[d_model, d_model] = [tokens, d_model]
    status = cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                d_model, total_tokens, d_model,
                &alpha,
                W_q, d_model,
                input, d_model,
                &beta,
                q_out, d_model);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[GQAProjection] Q projection cublasSgemm failed: status=" + std::to_string(status));
    }
    
    // K projection: [tokens, d_model] @ W_k^T[kv_dim, d_model] = [tokens, kv_dim]
    status = cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                kv_dim, total_tokens, d_model,
                &alpha,
                W_k, d_model,
                input, d_model,
                &beta,
                k_out, kv_dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[GQAProjection] K projection cublasSgemm failed: status=" + std::to_string(status));
    }
    
    // K-TRACE: After K projection (before bias)
    traceKTensor(k_out, total_tokens * kv_dim, "GQAProjection:K_matmul", -1, config.stream);
    
    // V projection: [tokens, d_model] @ W_v^T[kv_dim, d_model] = [tokens, kv_dim]
    status = cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                kv_dim, total_tokens, d_model,
                &alpha,
                W_v, d_model,
                input, d_model,
                &beta,
                v_out, kv_dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[GQAProjection] V projection cublasSgemm failed: status=" + std::to_string(status));
    }
    
    // Add biases
    if (b_q) addBias(q_out, b_q, total_tokens, d_model, config.stream);
    if (b_k) {
        addBias(k_out, b_k, total_tokens, kv_dim, config.stream);
        // K-TRACE: After K bias
        traceKTensor(k_out, total_tokens * kv_dim, "GQAProjection:K_bias", -1, config.stream);
    }
    if (b_v) addBias(v_out, b_v, total_tokens, kv_dim, config.stream);
    
    // Rule 22: Do NOT destroy handle - we don't own it!
}

// GQA-aware reshape: Q uses num_heads, K/V use num_kv_heads
//
// ASSUMPTIONS (caller must ensure):
//   - q_in: [tokens, d_model] where d_model = num_heads * head_dim, contiguous
//   - k_in: [tokens, kv_dim] where kv_dim = num_kv_heads * head_dim, contiguous
//   - v_in: [tokens, kv_dim] same layout as k_in
//   - Heads are packed contiguously within each token's features
//   - num_heads % num_kv_heads == 0 (for GQA head group alignment)
//
void launchQKVReshapeToBHSD(const float* q_in,
                            const float* k_in,
                            const float* v_in,
                            float* q_out,
                            float* k_out,
                            float* v_out,
                            int batch_size,
                            int seq_len,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            cudaStream_t stream) {
    // Rule 20: Fail loud on null pointers
    if (!q_in) throw std::runtime_error("[QKVReshapeToBHSD] q_in is NULL");
    if (!k_in) throw std::runtime_error("[QKVReshapeToBHSD] k_in is NULL");
    if (!v_in) throw std::runtime_error("[QKVReshapeToBHSD] v_in is NULL");
    if (!q_out) throw std::runtime_error("[QKVReshapeToBHSD] q_out is NULL");
    if (!k_out) throw std::runtime_error("[QKVReshapeToBHSD] k_out is NULL");
    if (!v_out) throw std::runtime_error("[QKVReshapeToBHSD] v_out is NULL");
    
    // Rule 20: Fail loud on invalid dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || 
        num_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("[QKVReshapeToBHSD] Invalid dimensions batch=" +
                std::to_string(batch_size) + " seq=" + std::to_string(seq_len) +
                " heads=" + std::to_string(num_heads) + " kv_heads=" + std::to_string(num_kv_heads) +
                " head_dim=" + std::to_string(head_dim));
    }
    
    // GQA alignment check
    if (num_kv_heads > num_heads || num_heads % num_kv_heads != 0) {
        throw std::runtime_error("[QKVReshapeToBHSD] Invalid GQA config num_heads=" +
                std::to_string(num_heads) + " must be divisible by num_kv_heads=" +
                std::to_string(num_kv_heads));
    }
    
    constexpr int threads = 256;
    cudaError_t err;
    
    // Q reshape: [tokens, d_model] -> [batch, num_heads, seq, head_dim]
    {
        const int total = batch_size * num_heads * seq_len * head_dim;
        const int blocks = (total + threads - 1) / threads;
        const int q_stride = num_heads * head_dim;  // d_model for Q
        reshapeQKVToBHSDKernel<<<blocks, threads, 0, stream>>>(
            q_in, q_out, batch_size, seq_len, num_heads, head_dim, q_stride);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[QKVReshapeToBHSD] Q reshape kernel failed: ") +
                    cudaGetErrorString(err));
        }
    }
    
    // K/V reshape: [tokens, kv_dim] -> [batch, num_kv_heads, seq, head_dim]
    // Use num_kv_heads instead of num_heads for GQA
    {
        const int total = batch_size * num_kv_heads * seq_len * head_dim;
        const int blocks = (total + threads - 1) / threads;
        const int kv_stride = num_kv_heads * head_dim;  // kv_dim for K/V
        reshapeQKVToBHSDKernel<<<blocks, threads, 0, stream>>>(
            k_in, k_out, batch_size, seq_len, num_kv_heads, head_dim, kv_stride);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[QKVReshapeToBHSD] K reshape kernel failed: ") +
                    cudaGetErrorString(err));
        }
        
        // K-TRACE: After K reshape to BHSD
        traceKTensor(k_out, total, "QKVReshapeToBHSD:K_bhsd", -1, stream);
        
        reshapeQKVToBHSDKernel<<<blocks, threads, 0, stream>>>(
            v_in, v_out, batch_size, seq_len, num_kv_heads, head_dim, kv_stride);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[QKVReshapeToBHSD] V reshape kernel failed: ") +
                    cudaGetErrorString(err));
        }
    }
}

void launchReshapeToBHSD(const float* src,
                         float* dst,
                         int batch_size,
                         int seq_len,
                         int num_heads,
                         int head_dim,
                         cudaStream_t stream) {
    // Rule 20: Fail loud on null pointers
    if (!src) throw std::runtime_error("[ReshapeToBHSD] src is NULL");
    if (!dst) throw std::runtime_error("[ReshapeToBHSD] dst is NULL");
    
    // Rule 20: Fail loud on invalid dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("[ReshapeToBHSD] Invalid dimensions batch=" +
                std::to_string(batch_size) + " seq=" + std::to_string(seq_len) +
                " heads=" + std::to_string(num_heads) + " head_dim=" + std::to_string(head_dim));
    }

    const int total = batch_size * num_heads * seq_len * head_dim;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const int src_stride = num_heads * head_dim;  // d_model

    // Reuse existing kernel: [tokens, src_stride] -> [batch, heads, seq, head_dim]
    reshapeQKVToBHSDKernel<<<blocks, threads, 0, stream>>>(
        src, dst, batch_size, seq_len, num_heads, head_dim, src_stride);
    
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[ReshapeToBHSD] kernel launch failed: ") +
                cudaGetErrorString(err));
    }
}

void launchReshapeFromBHSD(const float* src,
                           float* dst,
                           int batch_size,
                           int seq_len,
                           int num_heads,
                           int head_dim,
                           cudaStream_t stream) {
    // Rule 20: Fail loud on null pointers
    if (!src) throw std::runtime_error("[ReshapeFromBHSD] src is NULL");
    if (!dst) throw std::runtime_error("[ReshapeFromBHSD] dst is NULL");
    
    // Rule 20: Fail loud on invalid dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("[ReshapeFromBHSD] Invalid dimensions batch=" +
                std::to_string(batch_size) + " seq=" + std::to_string(seq_len) +
                " heads=" + std::to_string(num_heads) + " head_dim=" + std::to_string(head_dim));
    }

    const int d_model = num_heads * head_dim;
    const int total = batch_size * seq_len * d_model;
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    reshapeFromBHSDKernel<<<blocks, threads, 0, stream>>>(
        src, dst, batch_size, seq_len, num_heads, head_dim);
    
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[ReshapeFromBHSD] kernel launch failed: ") +
                cudaGetErrorString(err));
    }
}

} // namespace GRIM
