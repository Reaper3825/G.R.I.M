/**
 * @file Embedding_GPU.cu
 * @brief GPU-accelerated embedding: forward, backward, and runtime
 *
 * CONSOLIDATED from: Embedding_GPU.cu, EmbeddingBackward.cu, EmbeddingRuntime.cu
 *
 * MEMORY LAYOUT:
 * - token_embeddings: [vocab_size, d_model] row-major
 * - position_embeddings: [max_position, d_model] row-major
 * - output: [total_tokens, d_model] row-major
 *
 * KERNELS:
 * - EmbeddingLookupKernel: token + position lookup
 * - EmbeddingRMSNormKernel: fused lookup + RMSNorm (warp-level shuffle reduction)
 * - EmbeddingBackwardKernel: atomic scatter-add gradients
 *
 * WARP-LEVEL OPTIMIZATION:
 * - Uses 32-thread blocks (one warp) for RMSNorm reduction
 * - Warp shuffle intrinsics (__shfl_down_sync, __shfl_sync) replace shared memory
 * - No __syncthreads() needed - warp executes in lockstep
 * - Benefits: lower latency, no bank conflicts, higher SM occupancy
 */

#include "Embedding_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace GRIM {

//======================================================//
// Constants
//======================================================//

static constexpr int kEmbeddingBlockSize = 32;  // One warp - enables warp shuffle reduction

//======================================================//
// Device Helpers
//======================================================//

namespace {

/**
 * @brief Warp-level sum reduction using shuffle intrinsics
 * No __syncthreads() needed - warp executes in lockstep
 */
__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline int clampIndex(int value, int limit) {
    if (limit <= 0) return 0;
    if (value < 0) return 0;
    if (value >= limit) return limit - 1;
    return value;
}

//======================================================//
// Token ID Validation Kernel (Rule 20: Fail Loud)
//======================================================//

/**
 * @brief Validates token IDs and reports invalid entries
 * 
 * Sets error flags in device memory for host to check:
 * - error_flags[0]: count of invalid token IDs
 * - error_flags[1]: first invalid token index
 * - error_flags[2]: first invalid token value
 * 
 * @note This kernel should be called in debug builds or when validation is explicitly enabled
 */
__global__ void ValidateTokenIdsKernel(const int* token_ids,
                                       int total_tokens,
                                       int vocab_size,
                                       int* error_flags) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;
    
    const int token_id = token_ids[idx];
    
    if (token_id < 0 || token_id >= vocab_size) {
        // Atomically increment error count
        int prev_count = atomicAdd(&error_flags[0], 1);
        
        // Record first invalid token (only if we're the first to find one)
        if (prev_count == 0) {
            error_flags[1] = idx;
            error_flags[2] = token_id;
        }
    }
}

} // end anonymous namespace

/**
 * @brief Validates token IDs and throws if invalid tokens are found
 * 
 * Rule 20 (Fail Loud): Invalid token IDs should crash with clear error message
 * rather than silently producing garbage output.
 * 
 * @param token_ids Device pointer to token IDs
 * @param total_tokens Number of tokens to validate
 * @param vocab_size Valid token range is [0, vocab_size)
 * @param stream CUDA stream to use
 * @throws std::runtime_error if any token ID is negative or >= vocab_size
 */
void validateTokenIds(const int* token_ids,
                      int total_tokens,
                      int vocab_size,
                      cudaStream_t stream) {
    if (!token_ids || total_tokens <= 0) return;
    
    // Allocate error flags on device: [count, first_idx, first_value]
    int* d_error_flags = nullptr;
    cudaMallocAsync(&d_error_flags, 3 * sizeof(int), stream);
    cudaMemsetAsync(d_error_flags, 0, 3 * sizeof(int), stream);
    
    constexpr int kBlockSize = 256;
    const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
    
    ValidateTokenIdsKernel<<<grid, kBlockSize, 0, stream>>>(
        token_ids, total_tokens, vocab_size, d_error_flags);
    
    // Copy results back to host
    int h_error_flags[3] = {0, 0, 0};
    cudaMemcpyAsync(h_error_flags, d_error_flags, 3 * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    cudaFreeAsync(d_error_flags, stream);
    
    if (h_error_flags[0] > 0) {
        // Rule 20: Fail loud with detailed error message
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                 "validateTokenIds: Found %d invalid token ID(s). "
                 "First invalid at index %d: token_id=%d (valid range: [0, %d))",
                 h_error_flags[0], h_error_flags[1], h_error_flags[2], vocab_size);
        throw std::runtime_error(error_msg);
    }
}

namespace {

//======================================================//
// Forward Kernels
//======================================================//

/**
 * @brief Embedding lookup: token + position embeddings
 *
 * Grid: (total_tokens), Block: (256)
 * Position calculation when positions=nullptr: pos_id = token_idx % seq_len
 */
__global__ void EmbeddingLookupKernel(const int* token_ids,
                                      const int* positions,
                                      const float* token_embeddings,
                                      const float* position_embeddings,
                                      float* output,
                                      int total_tokens,
                                      int seq_len,
                                      int d_model,
                                      int vocab_size,
                                      int max_position) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int tid = threadIdx.x;
    const int token_id = token_ids ? token_ids[token_idx] : 0;

    if (token_id < 0 || token_id >= vocab_size) return;

    const int raw_pos = positions ? positions[token_idx] : (token_idx % seq_len);
    const int pos_id = clampIndex(raw_pos, max_position);

    const float* token_row = token_embeddings + static_cast<size_t>(token_id) * d_model;
    const float* pos_row = position_embeddings
                               ? position_embeddings + static_cast<size_t>(pos_id) * d_model
                               : nullptr;
    float* dst = output + static_cast<size_t>(token_idx) * d_model;

    for (int i = tid; i < d_model; i += blockDim.x) {
        float value = token_row[i];
        if (pos_row) value += pos_row[i];
        dst[i] = value;
    }
}

/**
 * @brief Fused embedding lookup + RMSNorm
 *
 * Grid: (total_tokens), Block: (32) - One warp for efficient shuffle reduction
 */
__global__ void EmbeddingRMSNormKernel(const int* token_ids,
                                       const int* positions,
                                       const float* token_embeddings,
                                       const float* position_embeddings,
                                       const float* gamma,
                                       float* output,
                                       int total_tokens,
                                       int seq_len,
                                       int d_model,
                                       int vocab_size,
                                       int max_position,
                                       float eps) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const int tid = threadIdx.x;
    const int token_id = token_ids ? token_ids[token_idx] : 0;

    if (token_id < 0 || token_id >= vocab_size) return;

    const int raw_pos = positions ? positions[token_idx] : (token_idx % seq_len);
    const int pos_id = clampIndex(raw_pos, max_position);

    float* out_ptr = output + static_cast<size_t>(token_idx) * d_model;
    const float* token_ptr = token_embeddings + static_cast<size_t>(token_id) * d_model;
    const float* pos_ptr = position_embeddings
                               ? position_embeddings + static_cast<size_t>(pos_id) * d_model
                               : nullptr;

    // Pass 1: compute sum of squares (each thread handles multiple elements)
    float sq_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        float value = token_ptr[i];
        if (pos_ptr) value += pos_ptr[i];
        out_ptr[i] = value;
        sq_sum += value * value;
    }

    // Warp-level reduction - no __syncthreads needed
    sq_sum = warpReduceSum(sq_sum);

    // Broadcast inv_rms from lane 0 to all threads
    float inv_rms = rsqrtf(sq_sum / static_cast<float>(d_model) + eps);
    inv_rms = __shfl_sync(0xffffffff, inv_rms, 0);

    // Pass 2: normalize
    for (int i = tid; i < d_model; i += blockDim.x) {
        float normalized = out_ptr[i] * inv_rms;
        float scale = gamma ? gamma[i] : 1.0f;
        out_ptr[i] = normalized * scale;
    }
}

//======================================================//
// Backward Kernel
//======================================================//

__global__ void EmbeddingBackwardKernel(const float* __restrict__ grad_output,
                                        const int* __restrict__ token_ids,
                                        float* __restrict__ grad_embeddings,
                                        int total_tokens,
                                        int d_model,
                                        int vocab_size) {
    const int token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= total_tokens) return;

    const int token_id = token_ids ? token_ids[token_index] : -1;
    if (token_id < 0 || token_id >= vocab_size) return;

    const float* grad_src = grad_output + static_cast<size_t>(token_index) * d_model;
    float* grad_dst = grad_embeddings + static_cast<size_t>(token_id) * d_model;

    for (int i = 0; i < d_model; ++i) {
        atomicAdd(&grad_dst[i], grad_src[i]);
    }
}

} // anonymous namespace

//======================================================//
// Forward Kernel Launcher
//======================================================//

void launchEmbeddingLookup(const EmbeddingForwardArgs& args,
                           const EmbeddingConfig& config) {
    // Rule 20: Crash on invalid input
    args.validate("launchEmbeddingLookup");
    
    // Extract pointers from TensorViews
    float* output_ptr = args.output.ptr;
    const float* token_embeddings_ptr = args.weights->token_embeddings.ptr;
    const float* position_embeddings_ptr = args.weights->position_embeddings.ptr;
    const float* gamma_ptr = args.weights->gamma.ptr;

    const int total_tokens = args.batch_size * args.seq_len;
    if (total_tokens <= 0) {
        throw std::runtime_error("launchEmbeddingLookup: total_tokens=" + std::to_string(total_tokens));
    }
    if (config.d_model <= 0) {
        throw std::runtime_error("launchEmbeddingLookup: d_model=" + std::to_string(config.d_model));
    }
    if (config.vocab_size <= 0) {
        throw std::runtime_error("launchEmbeddingLookup: vocab_size=" + std::to_string(config.vocab_size));
    }
    if (config.max_position <= 0 && position_embeddings_ptr) {
        throw std::runtime_error("launchEmbeddingLookup: position_embeddings provided but max_position=0");
    }

    cudaStream_t stream = args.stream ? args.stream : config.stream;
    if (!stream) {
        fprintf(stderr, "[FATAL] launchEmbeddingLookup: stream is NULL (default stream usage disallowed)\n");
        throw std::runtime_error("launchEmbeddingLookup: stream is NULL");
    }

    // Rule 20: Validate token IDs in debug builds to catch data pipeline bugs early
    bool debug_tid_validation = true;
    if (debug_tid_validation) {
        validateTokenIds(args.token_ids, total_tokens, config.vocab_size, stream);
    }



    const int seq_len = (args.seq_len > 0) ? args.seq_len : total_tokens;
    dim3 block(kEmbeddingBlockSize);
    dim3 grid(total_tokens);

    if (config.apply_rms_norm && gamma_ptr) {
        EmbeddingRMSNormKernel<<<grid, block, 0, stream>>>(
            args.token_ids, args.positions,
            token_embeddings_ptr, position_embeddings_ptr,
            gamma_ptr, output_ptr,
            total_tokens, seq_len, config.d_model,
            config.vocab_size, config.max_position, config.rms_epsilon);
    } else {
        EmbeddingLookupKernel<<<grid, block, 0, stream>>>(
            args.token_ids, args.positions,
            token_embeddings_ptr, position_embeddings_ptr,
            output_ptr, total_tokens, seq_len, config.d_model,
            config.vocab_size, config.max_position);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Embedding kernel failed: " + std::string(cudaGetErrorString(err)));
    }
}

//======================================================//
// Backward Kernel Launcher
//======================================================//

void launchEmbeddingBackward(const float* grad_output,
                             const int* token_ids,
                             float* grad_embeddings,
                             int batch_size,
                             int seq_len,
                             int d_model,
                             int vocab_size,
                             cudaStream_t stream) {
    // Rule 20: Crash on invalid input
    if (!grad_output) {
        throw std::runtime_error("launchEmbeddingBackward: grad_output is NULL");
    }
    if (!token_ids) {
        throw std::runtime_error("launchEmbeddingBackward: token_ids is NULL");
    }
    if (!grad_embeddings) {
        throw std::runtime_error("launchEmbeddingBackward: grad_embeddings is NULL");
    }

    const int total_tokens = batch_size * seq_len;
    if (total_tokens <= 0) {
        throw std::runtime_error("launchEmbeddingBackward: total_tokens=" + std::to_string(total_tokens));
    }
    if (d_model <= 0) {
        throw std::runtime_error("launchEmbeddingBackward: d_model=" + std::to_string(d_model));
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("launchEmbeddingBackward: vocab_size=" + std::to_string(vocab_size));
    }

    // Rule 20: Validate token IDs in debug builds to catch data pipeline bugs early
#ifndef NDEBUG
    validateTokenIds(token_ids, total_tokens, vocab_size, stream);
#endif

    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
    EmbeddingBackwardKernel<<<grid, kBlockSize, 0, stream>>>(
        grad_output, token_ids, grad_embeddings, total_tokens, d_model, vocab_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("EmbeddingBackward kernel failed: " + std::string(cudaGetErrorString(err)));
    }
}

//======================================================//
// Position Embedding Backward Kernel (Issue #36 FIX)
//======================================================//

/**
 * Position embedding backward: accumulates gradients from all batch elements
 * for each position index. Unlike token embedding backward (which scatters by 
 * token_id), this scatters by position index (0 to seq_len-1).
 *
 * For position p in batch element b:
 *   grad_position_embeddings[p] += grad_output[b * seq_len + p]
 *
 * This matches PyTorch nn.Embedding behavior for position embeddings.
 */
__global__ void PositionEmbeddingBackwardKernel(const float* __restrict__ grad_output,
                                                float* __restrict__ grad_position_embeddings,
                                                int total_tokens,
                                                int seq_len,
                                                int d_model,
                                                int max_seq_len) {
    const int token_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_index >= total_tokens) return;

    // Compute position index within sequence (0 to seq_len-1)
    const int pos_id = token_index % seq_len;
    if (pos_id < 0 || pos_id >= max_seq_len) return;

    const float* grad_src = grad_output + static_cast<size_t>(token_index) * d_model;
    float* grad_dst = grad_position_embeddings + static_cast<size_t>(pos_id) * d_model;

    for (int i = 0; i < d_model; ++i) {
        atomicAdd(&grad_dst[i], grad_src[i]);
    }
}

void launchPositionEmbeddingBackward(const float* grad_output,
                                     float* grad_position_embeddings,
                                     int batch_size,
                                     int seq_len,
                                     int d_model,
                                     int max_seq_len,
                                     cudaStream_t stream) {
    // Rule 20: Crash on invalid input
    if (!grad_output) {
        throw std::runtime_error("launchPositionEmbeddingBackward: grad_output is NULL");
    }
    if (!grad_position_embeddings) {
        throw std::runtime_error("launchPositionEmbeddingBackward: grad_position_embeddings is NULL");
    }

    const int total_tokens = batch_size * seq_len;
    if (total_tokens <= 0) {
        throw std::runtime_error("launchPositionEmbeddingBackward: total_tokens=" + std::to_string(total_tokens));
    }
    if (d_model <= 0) {
        throw std::runtime_error("launchPositionEmbeddingBackward: d_model=" + std::to_string(d_model));
    }
    if (seq_len <= 0 || seq_len > max_seq_len) {
        throw std::runtime_error("launchPositionEmbeddingBackward: seq_len=" + std::to_string(seq_len) 
                                 + " max_seq_len=" + std::to_string(max_seq_len));
    }

    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
    PositionEmbeddingBackwardKernel<<<grid, kBlockSize, 0, stream>>>(
        grad_output, grad_position_embeddings, total_tokens, seq_len, d_model, max_seq_len);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("PositionEmbeddingBackward kernel failed: " + std::string(cudaGetErrorString(err)));
    }
}

//======================================================//
// EmbeddingLayer
//======================================================//

void EmbeddingLayer::forward(const EmbeddingForwardArgs& args) {
    if (!args.weights) {
        throw std::runtime_error("EmbeddingLayer::forward: args.weights is NULL");
    }
    if (!args.stream && !config_.stream) {
        fprintf(stderr, "[FATAL] EmbeddingLayer::forward: stream is NULL (default stream usage disallowed)\n");
        throw std::runtime_error("EmbeddingLayer::forward: stream is NULL");
    }

    EmbeddingForwardArgs call_args = args;
    if (!call_args.stream) call_args.stream = config_.stream;
    launchEmbeddingLookup(call_args, config_);
}

//======================================================//
// Runtime Validation Helpers
//======================================================//

static bool validateRuntime(const EmbeddingRuntime* runtime, const char* caller) {
    if (!runtime) {
        fprintf(stderr, "[%s] ERROR: EmbeddingRuntime is null\n", caller);
        return false;
    }
    if (!runtime->weights.token_embeddings.ptr) {
        fprintf(stderr, "[%s] ERROR: token_embeddings is null\n", caller);
        return false;
    }
    if (runtime->config.vocab_size <= 0) {
        fprintf(stderr, "[%s] ERROR: vocab_size=%d invalid\n", caller, runtime->config.vocab_size);
        return false;
    }
    if (runtime->config.d_model <= 0) {
        fprintf(stderr, "[%s] ERROR: d_model=%d invalid\n", caller, runtime->config.d_model);
        return false;
    }
    return true;
}

static bool validateForwardArgs(int batch_size, int seq_len, float* output, const char* caller) {
    if (batch_size <= 0) {
        fprintf(stderr, "[%s] ERROR: batch_size=%d invalid\n", caller, batch_size);
        return false;
    }
    if (seq_len <= 0) {
        fprintf(stderr, "[%s] ERROR: seq_len=%d invalid\n", caller, seq_len);
        return false;
    }
    if (!output) {
        fprintf(stderr, "[%s] ERROR: output is null\n", caller);
        return false;
    }
    return true;
}

//======================================================//
// Runtime Lifecycle
//======================================================//

void destroyEmbeddingRuntime(EmbeddingRuntime* runtime) {
    if (!runtime) return;

    if (runtime->token_buffer) { cudaFree(runtime->token_buffer); runtime->token_buffer = nullptr; }
    if (runtime->position_buffer) { cudaFree(runtime->position_buffer); runtime->position_buffer = nullptr; }
    if (runtime->gamma_buffer) { cudaFree(runtime->gamma_buffer); runtime->gamma_buffer = nullptr; }
    if (runtime->single_token_id) { cudaFree(runtime->single_token_id); runtime->single_token_id = nullptr; }
    if (runtime->single_position) { cudaFree(runtime->single_position); runtime->single_position = nullptr; }
    if (runtime->owns_stream && runtime->stream) { cudaStreamDestroy(runtime->stream); runtime->stream = nullptr; }

    delete runtime;
}

//======================================================//
// Runtime Forward - Batched
//======================================================//

bool embeddingRuntimeForward(EmbeddingRuntime* runtime,
                             const int* token_ids,
                             const int* positions,
                             int batch_size,
                             int seq_len,
                             float* output) {
    if (!validateRuntime(runtime, "embeddingRuntimeForward")) return false;
    if (!validateForwardArgs(batch_size, seq_len, output, "embeddingRuntimeForward")) return false;

    if (seq_len > runtime->config.max_position && runtime->weights.position_embeddings.ptr) {
        fprintf(stderr, "[embeddingRuntimeForward] WARNING: seq_len=%d > max_position=%d\n",
                seq_len, runtime->config.max_position);
    }

    // Construct TensorView for output (BSM layout)
    const int total_tokens = batch_size * seq_len;
    TensorContract::TensorView output_view = TensorContract::TensorView::make_BSM(
        output, total_tokens, runtime->config.d_model, "embedding_output");

    EmbeddingForwardArgs args{};
    args.token_ids = token_ids;
    args.positions = positions;
    args.batch_size = batch_size;
    args.seq_len = seq_len;
    args.output = output_view;
    args.weights = &runtime->weights;
    args.stream = runtime->stream;

    launchEmbeddingLookup(args, runtime->config);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[embeddingRuntimeForward] Kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

//======================================================//
// Runtime Forward - Single Token
//======================================================//

bool embeddingRuntimeForwardSingle(EmbeddingRuntime* runtime,
                                   int token_id,
                                   int position,
                                   float* output) {
    if (!validateRuntime(runtime, "embeddingRuntimeForwardSingle")) return false;
    if (!output) {
        fprintf(stderr, "[embeddingRuntimeForwardSingle] ERROR: output is null\n");
        return false;
    }

    if (token_id < 0 || token_id >= runtime->config.vocab_size) {
        fprintf(stderr, "[embeddingRuntimeForwardSingle] ERROR: token_id=%d out of range [0, %d)\n",
                token_id, runtime->config.vocab_size);
        return false;
    }

    if (position < 0) {
        fprintf(stderr, "[embeddingRuntimeForwardSingle] ERROR: position=%d negative\n", position);
        return false;
    }
    if (position >= runtime->config.max_position && runtime->weights.position_embeddings.ptr) {
        fprintf(stderr, "[embeddingRuntimeForwardSingle] WARNING: position=%d >= max_position=%d\n",
                position, runtime->config.max_position);
    }

    cudaMemcpyAsync(runtime->single_token_id, &token_id, sizeof(int),
                    cudaMemcpyHostToDevice, runtime->stream);
    cudaMemcpyAsync(runtime->single_position, &position, sizeof(int),
                    cudaMemcpyHostToDevice, runtime->stream);

    // Construct TensorView for single-token output (BSM layout [1, d_model])
    TensorContract::TensorView output_view = TensorContract::TensorView::make_BSM(
        output, 1, runtime->config.d_model, "embedding_single_output");

    EmbeddingForwardArgs args{};
    args.token_ids = runtime->single_token_id;
    args.positions = runtime->single_position;
    args.batch_size = 1;
    args.seq_len = 1;
    args.output = output_view;
    args.weights = &runtime->weights;
    args.stream = runtime->stream;

    launchEmbeddingLookup(args, runtime->config);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[embeddingRuntimeForwardSingle] Kernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace GRIM
