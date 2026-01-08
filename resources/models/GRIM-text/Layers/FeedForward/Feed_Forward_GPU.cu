//======================================================//
//  Feed_Forward_GPU.cu
//  GPU-accelerated FeedForward layer implementation
//  
//  Two-layer MLP with GELU activation:
//    hidden = GELU(input @ W1^T + b1)
//    output = hidden @ W2^T + b2
//  
//  Uses shared components:
//    - Shared/Activations/GELU for GELU forward/backward
//======================================================//

#define USE_CUDA

#include "Feed_Forward_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

namespace GRIM {

//======================================================//
//  FFN-specific CUDA Kernels (Bias operations)
//======================================================//

namespace {

__global__ void ffnBiasAddKernel(float* __restrict__ tensor,
                                  const float* __restrict__ bias,
                                  int total_elements,
                                  int features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const int feature_idx = idx % features;
        tensor[idx] += bias[feature_idx];
    }
}

// Optimized bias backward using shared memory reduction
__global__ void ffnBiasBackwardKernel(const float* __restrict__ grad_output,
                                       float* __restrict__ grad_bias,
                                       int total_tokens,
                                       int features) {
    extern __shared__ float sdata[];
    
    const int feature_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = blockDim.x;
    
    if (feature_idx >= features) return;
    
    // Each thread sums multiple tokens
    float local_sum = 0.0f;
    for (int t = tid; t < total_tokens; t += grid_stride) {
        local_sum += grad_output[t * features + feature_idx];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&grad_bias[feature_idx], sdata[0]);
    }
}

} // anonymous namespace

//======================================================//
//  FFN Kernel Launch Functions
//======================================================//

void launchFFNBiasAdd(float* tensor, const float* bias,
                      int total_tokens, int features,
                      cudaStream_t stream) {
    // Rule 20: Crash on invalid input - caller MUST provide valid pointers
    if (!tensor) {
        throw std::runtime_error("launchFFNBiasAdd: tensor is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!bias) {
        throw std::runtime_error("launchFFNBiasAdd: bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (total_tokens <= 0 || features <= 0) {
        throw std::runtime_error("launchFFNBiasAdd: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");
    }
    const int total_elements = total_tokens * features;
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = (total_elements + kBlockSize - 1) / kBlockSize;
    ffnBiasAddKernel<<<grid, kBlockSize, 0, stream>>>(tensor, bias, total_elements, features);
}

void launchFFNBiasBackward(const float* grad_output, float* grad_bias,
                           int total_tokens, int features,
                           cudaStream_t stream) {
    // Rule 20: Crash on invalid input - caller MUST provide valid pointers
    if (!grad_output) {
        throw std::runtime_error("launchFFNBiasBackward: grad_output is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!grad_bias) {
        throw std::runtime_error("launchFFNBiasBackward: grad_bias is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (total_tokens <= 0 || features <= 0) {
        throw std::runtime_error("launchFFNBiasBackward: invalid dimensions (" + std::to_string(total_tokens) + ", " + std::to_string(features) + ")");
    }
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int shared_bytes = kBlockSize * sizeof(float);
    ffnBiasBackwardKernel<<<features, kBlockSize, shared_bytes, stream>>>(
        grad_output, grad_bias, total_tokens, features);
}

//======================================================//
//  FeedForwardLayer Implementation
//======================================================//

FeedForwardLayer::FeedForwardLayer() {
    initCublas();
}

FeedForwardLayer::FeedForwardLayer(const FeedForwardConfig& config)
    : config_(config) {
    initCublas();
}

FeedForwardLayer::FeedForwardLayer(const Dimensions& dims,
                                   const FeedForwardConfig& config)
    : Layer(dims), config_(config) {
    initCublas();
}

FeedForwardLayer::~FeedForwardLayer() {
    destroyCublas();
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : Layer(std::move(other))
    , config_(other.config_)
    , W1_(std::move(other.W1_))
    , b1_(std::move(other.b1_))
    , W2_(std::move(other.W2_))
    , b2_(std::move(other.b2_))
    , pre_gelu_buf_(std::move(other.pre_gelu_buf_))
    , post_gelu_buf_(std::move(other.post_gelu_buf_)) {
    other.config_.cublas_handle = nullptr;
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        destroyCublas();
        Layer::operator=(std::move(other));
        config_ = other.config_;
        config_.cublas_handle = other.config_.cublas_handle;
        W1_ = std::move(other.W1_);
        b1_ = std::move(other.b1_);
        W2_ = std::move(other.W2_);
        b2_ = std::move(other.b2_);
        pre_gelu_buf_ = std::move(other.pre_gelu_buf_);
        post_gelu_buf_ = std::move(other.post_gelu_buf_);
        other.config_.cublas_handle = nullptr;
    }
    return *this;
}

void FeedForwardLayer::initCublas() {
    // Use centralized cuBLAS handle (Rule 22: NO local handle creation)
    if (!config_.cublas_handle) {
        throw std::runtime_error("FeedForwardLayer::initCublas: config_.cublas_handle is NULL. "
                                 "MUST pass training_state.cublas_handle per Rule 22.");
    }
    config_.cublas_handle = config_.cublas_handle;  // Store pointer, do NOT own
}

void FeedForwardLayer::destroyCublas() {
    // Rule 22: Do NOT destroy config_.cublas_handle - we don't own it!
    // It's owned by training_state and destroyed in TrainingState destructor
    config_.cublas_handle = nullptr;  // Just clear pointer
}

void FeedForwardLayer::setConfig(const FeedForwardConfig& cfg) {
    config_ = cfg;
    // REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu
}

void FeedForwardLayer::ensureWeightStorage() {
    if (config_.d_model <= 0 || config_.d_ff <= 0) {
        throw std::runtime_error("FeedForwardLayer: d_model and d_ff must be positive");
    }
    StreamController::fatalIfDefaultStream(config_.stream, "FeedForwardLayer::ensureWeightStorage");
    
    const std::size_t w1_size = static_cast<std::size_t>(config_.d_ff) * config_.d_model;
    const std::size_t w2_size = static_cast<std::size_t>(config_.d_model) * config_.d_ff;
    
    if (W1_.size() != w1_size) {
        W1_.allocate(w1_size);
    }
    if (b1_.size() != static_cast<std::size_t>(config_.d_ff)) {
        b1_.allocate(config_.d_ff);
        cudaMemsetAsync(b1_.ptr(), 0, config_.d_ff * sizeof(float), config_.stream);
    }
    if (W2_.size() != w2_size) {
        W2_.allocate(w2_size);
    }
    if (b2_.size() != static_cast<std::size_t>(config_.d_model)) {
        b2_.allocate(config_.d_model);
        cudaMemsetAsync(b2_.ptr(), 0, config_.d_model * sizeof(float), config_.stream);
    }
}

// DELETED: resolveStream() - Rule 20
// Stream fallback hides bugs. Caller MUST always provide stream explicitly.
// If config_.stream is needed, caller should pass it directly.

std::size_t FeedForwardLayer::requiredWorkspaceBytes(int total_tokens) const {
    if (total_tokens <= 0) return 0;
    const std::size_t tokens = static_cast<std::size_t>(total_tokens);
    const std::size_t d_ff = static_cast<std::size_t>(config_.d_ff);
    // pre_gelu + post_gelu + grad_hidden (for backward)
    return tokens * d_ff * 3 * sizeof(float);
}

//======================================================//
//  Forward Pass
//======================================================//

void FeedForwardLayer::forward(const FeedForwardForwardArgs& args,
                               LayerWorkspace<float>* /*workspace*/) {
    forwardGPU(args.input, args.output, args.total_tokens,
               args.cache_pre_gelu, args.cache_post_gelu);
}

void FeedForwardLayer::forwardGPU(const float* d_input, float* d_output,
                                  int total_tokens,
                                  float* d_pre_gelu_cache,
                                  float* d_post_gelu_workspace) {
    // Rule 20: Crash on invalid input - caller MUST provide valid pointers
    if (!d_input) {
        throw std::runtime_error("FeedForwardLayer::forwardGPU: d_input is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!d_output) {
        throw std::runtime_error("FeedForwardLayer::forwardGPU: d_output is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (total_tokens <= 0) {
        throw std::runtime_error("FeedForwardLayer::forwardGPU: total_tokens <= 0 at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (!W1_.ptr() || !W2_.ptr()) {
        throw std::runtime_error("FeedForwardLayer::forwardGPU: weights not initialized");
    }
    if (!config_.stream) {
        fprintf(stderr, "[FATAL] FeedForwardLayer::forwardGPU: config_.stream is NULL (default stream usage disallowed)\n");
        throw std::runtime_error("FeedForwardLayer::forwardGPU: config_.stream is NULL - MUST provide stream (Rule 20)");
    }

    const cudaStream_t stream = config_.stream;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu

    // Allocate internal buffers if external ones not provided
    const std::size_t hidden_elements = static_cast<std::size_t>(total_tokens) * config_.d_ff;
    
    float* pre_gelu = d_pre_gelu_cache;
    if (!pre_gelu) {
        if (pre_gelu_buf_.size() < hidden_elements) {
            pre_gelu_buf_.allocate(hidden_elements);
        }
        pre_gelu = pre_gelu_buf_.ptr();
    }

    float* post_gelu = d_post_gelu_workspace;
    if (!post_gelu) {
        if (post_gelu_buf_.size() < hidden_elements) {
            post_gelu_buf_.allocate(hidden_elements);
        }
        post_gelu = post_gelu_buf_.ptr();
    }

    // Layer 1: pre_gelu = input @ W1^T + b1
    // ROW-MAJOR: [tokens, d_model] @ [d_model, d_ff] = [tokens, d_ff]
    // W1 stored as [d_ff, d_model] row-major, so we need W1^T
    // cuBLAS row-major: C[M,N] = A[M,K] @ B^T[K,N] where B is [N,K]
    // Here: M=tokens, N=d_ff, K=d_model
    cublasSgemm(config_.cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                config_.d_ff, total_tokens, config_.d_model,
                &alpha,
                W1_.ptr(), config_.d_model,    // W1 [d_ff, d_model], ldb=d_model
                d_input, config_.d_model,       // input [tokens, d_model], lda=d_model
                &beta,
                pre_gelu, config_.d_ff);        // output [tokens, d_ff], ldc=d_ff

    // Add bias b1
    launchFFNBiasAdd(pre_gelu, b1_.ptr(), total_tokens, config_.d_ff, stream);

    // Apply GELU using shared module
    GELUForwardArgs gelu_args{};
    gelu_args.input = pre_gelu;
    gelu_args.output = post_gelu;
    gelu_args.elements = hidden_elements;
    gelu_args.stream = stream;
    launchGeluForward(gelu_args);

    // Layer 2: output = post_gelu @ W2^T + b2
    // ROW-MAJOR: [tokens, d_ff] @ [d_ff, d_model] = [tokens, d_model]
    // W2 stored as [d_model, d_ff] row-major, so we need W2^T
    // Here: M=tokens, N=d_model, K=d_ff
    cublasSgemm(config_.cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                config_.d_model, total_tokens, config_.d_ff,
                &alpha,
                W2_.ptr(), config_.d_ff,        // W2 [d_model, d_ff], ldb=d_ff
                post_gelu, config_.d_ff,        // post_gelu [tokens, d_ff], lda=d_ff
                &beta,
                d_output, config_.d_model);     // output [tokens, d_model], ldc=d_model

    // Add bias b2
    launchFFNBiasAdd(d_output, b2_.ptr(), total_tokens, config_.d_model, stream);
}

//======================================================//
//  DELETED: backward(), backwardGPU() - Rule 20
//  Training uses BackwardPhase2_Encoder.cu::computeFFNBackward()
//  which implements FFN backward directly with proper gradient tracking.
//  This dead code was never called by production training.
//======================================================//

//======================================================//
//  DELETED: CRTP Interface - Rule 20
//  onConfigure, forwardImpl, backwardImpl, applyGradientsImpl removed.
//  Use forward(FeedForwardForwardArgs&) directly.
//  Training backward uses BackwardPhase2_Encoder.cu.
//======================================================//

} // namespace GRIM
