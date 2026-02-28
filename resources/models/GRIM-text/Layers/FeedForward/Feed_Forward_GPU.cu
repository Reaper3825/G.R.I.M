//======================================================//
//  Feed_Forward_GPU.cu
//  GPU-accelerated FeedForward layer using autograd
//
//  Two-layer MLP with GELU activation:
//    hidden = GELU(input @ W1 + b1)
//    output = hidden @ W2 + b2
//
//  ISSUE #97 FIX: Bias add now uses autograd::broadcast_add for proper
//  gradient tracking. Previously used raw kernels which bypassed autograd,
//  causing b1/b2 to receive ZERO gradients (frozen biases).
//======================================================//

#include "Feed_Forward_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"  // Issue #97: autograd::broadcast_add

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cstdio>

// Issue #142: In-place scaling kernel for residual projection init
static __global__ void kernel_ffn_scale_inplace(float* data, size_t count, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

namespace GRIM {

//======================================================//
//  FeedForwardLayer Implementation
//======================================================//

//--------------------------------------------------
// Self-allocating constructor (Pattern B: layer self-management)
// Follows ScratchBlockLayer pattern: allocate → ensure_grad → Xavier init
//--------------------------------------------------
FeedForwardLayer::FeedForwardLayer(const FeedForwardConfig& config, uint64_t seed, float residual_scale)
    : config_(config) {
    if (!config_.cublas_handle) {
        throw std::runtime_error("FeedForwardLayer: cublas_handle is NULL - caller MUST provide handle");
    }
    if (!config_.stream) {
        throw std::runtime_error("FeedForwardLayer: stream is NULL");
    }
    if (config_.d_model <= 0 || config_.d_ff <= 0) {
        throw std::runtime_error("FeedForwardLayer: d_model and d_ff must be positive");
    }
    
    const int d_model = config_.d_model;
    const int d_ff = config_.d_ff;
    const cudaStream_t stream = config_.stream;
    
    // W1: [d_model, d_ff] for matmul: input [T, d_model] @ W1 [d_model, d_ff] = [T, d_ff]
    W1_ = Tensor::zeros({d_model, d_ff}, stream, "ffn_w1");
    W1_.requires_grad_();
    W1_.ensure_grad();
    Tensor::xavier_uniform_(W1_, seed, stream);
    
    // W2: [d_ff, d_model] for matmul: hidden [T, d_ff] @ W2 [d_ff, d_model] = [T, d_model]
    // Issue #142: Scaled by residual_scale after Xavier init (GPT-2 pattern)
    W2_ = Tensor::zeros({d_ff, d_model}, stream, "ffn_w2");
    W2_.requires_grad_();
    W2_.ensure_grad();
    Tensor::xavier_uniform_(W2_, seed + 1, stream);
    
    // Apply Issue #142 residual scaling to W2 (down-projection)
    if (residual_scale != 1.0f) {
        const size_t count = W2_.numel();
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        kernel_ffn_scale_inplace<<<blocks, threads, 0, stream>>>(W2_.data, count, residual_scale);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("FeedForwardLayer: residual scale kernel failed: " + 
                                     std::string(cudaGetErrorString(err)));
        }
    }
    
    if (config_.use_bias) {
        b1_ = Tensor::zeros({1, d_ff}, stream, "ffn_b1");
        b1_.requires_grad_();
        b1_.ensure_grad();
        // Biases initialized to zero (standard practice)
        
        b2_ = Tensor::zeros({1, d_model}, stream, "ffn_b2");
        b2_.requires_grad_();
        b2_.ensure_grad();
    }
    
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    
    std::fprintf(stderr, "[FeedForwardLayer] Self-allocated weights: W1=[%d,%d], W2=[%d,%d], residual_scale=%.6f\n",
                 d_model, d_ff, d_ff, d_model, residual_scale);
}

FeedForwardLayer::~FeedForwardLayer() {
    // Tensors clean up via RAII
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : config_(other.config_)
    , W1_(std::move(other.W1_))
    , b1_(std::move(other.b1_))
    , W2_(std::move(other.W2_))
    , b2_(std::move(other.b2_)) {
    other.config_.cublas_handle = nullptr;
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        W1_ = std::move(other.W1_);
        b1_ = std::move(other.b1_);
        W2_ = std::move(other.W2_);
        b2_ = std::move(other.b2_);
        other.config_.cublas_handle = nullptr;
    }
    return *this;
}

void FeedForwardLayer::setConfig(const FeedForwardConfig& cfg) {
    config_ = cfg;
    if (config_.cublas_handle) {
        autograd::set_autograd_cublas_handle(config_.cublas_handle);
    }
}



//======================================================//
//  Forward Pass - Autograd with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor FeedForwardLayer::forward(const Tensor& input, ForwardIntermediates& intermediates,
                                 uint64_t training_step, int layer_idx) {
    // Rule 20: Crash on invalid weights
    if (!W1_.data || !W2_.data) {
        throw std::runtime_error("FeedForwardLayer::forward: weights not set - constructor requires external weights");
    }
    if (!config_.stream) {
        throw std::runtime_error("FeedForwardLayer::forward: stream is NULL");
    }

    const cudaStream_t stream = config_.stream;

    // Extract dimensions from input shape
    const auto& in_shape = input.shape.as_2d();
    const int total_tokens = in_shape.rows;
    const int d_model = in_shape.cols;

    if (d_model != config_.d_model) {
        throw std::runtime_error("FeedForwardLayer::forward: input d_model mismatch (" +
                                 std::to_string(d_model) + " vs " + std::to_string(config_.d_model) + ")");
    }

    // Sanity-check weight shapes (avoid silent wrong matmuls)
    {
        const auto& s1 = W1_.shape.as_2d();
        if (s1.rows != config_.d_model || s1.cols != config_.d_ff) {
            throw std::runtime_error("FeedForwardLayer::forward: W1 shape mismatch. Expected [" +
                                     std::to_string(config_.d_model) + "," + std::to_string(config_.d_ff) +
                                     "], got [" + std::to_string(s1.rows) + "," + std::to_string(s1.cols) + "]");
        }
    }
    {
        const auto& s2 = W2_.shape.as_2d();
        if (s2.rows != config_.d_ff || s2.cols != config_.d_model) {
            throw std::runtime_error("FeedForwardLayer::forward: W2 shape mismatch. Expected [" +
                                     std::to_string(config_.d_ff) + "," + std::to_string(config_.d_model) +
                                     "], got [" + std::to_string(s2.rows) + "," + std::to_string(s2.cols) + "]");
        }
    }

    // Set cuBLAS handle for autograd ops
    autograd::set_autograd_cublas_handle(config_.cublas_handle);

    //--------------------------------------------------
    // Layer 1: hidden = GELU(input @ W1 + b1)
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------

    // matmul: input [tokens, d_model] @ W1 [d_model, d_ff] = pre_gelu [tokens, d_ff]
    if (!input.data) {
        throw std::runtime_error("FeedForwardLayer::forward: input.data is NULL before W1 matmul (cannot supply required a_cache for W1 grad)");
    }
    intermediates.ffn_linear1_out = autograd::matmul(input, W1_, stream,
                                                     input.data,  // cache input for W1 grad
                                                     nullptr);    // W1 persists, no cache needed

    // ISSUE #97 FIX: Add bias b1 with autograd tracking (was bypassing gradient computation)
    if (config_.use_bias && b1_.data) {
        intermediates.ffn_linear1_out = autograd::broadcast_add(intermediates.ffn_linear1_out, b1_, stream);
    }

    // GELU activation - stores result in intermediates
    intermediates.ffn_gelu_out = autograd::gelu(intermediates.ffn_linear1_out, stream, 
                                                intermediates.ffn_linear1_out.data);

    // FFN activation dropout: applied after GELU, before W2 projection.
    // Standard practice (GPT-2, LLaMA, BERT): dropout(GELU(x @ W1 + b1)) @ W2 + b2
    if (config_.dropout_rate > 0.0f && training_step > 0) {
        const uint64_t ffn_act_dropout_seed = training_step * 2654435761ULL + 300 + layer_idx;
        intermediates.ffn_gelu_out = autograd::dropout(intermediates.ffn_gelu_out, config_.dropout_rate,
                                                       ffn_act_dropout_seed, true, stream);
    }

    //--------------------------------------------------
    // Layer 2: output = post_gelu @ W2 + b2
    // The output is also stored in intermediates.ffn_out by EncodingLayer
    //--------------------------------------------------

    // matmul: post_gelu [tokens, d_ff] @ W2 [d_ff, d_model] = output [tokens, d_model]
    if (!intermediates.ffn_gelu_out.data) {
        throw std::runtime_error("FeedForwardLayer::forward: ffn_gelu_out.data is NULL before W2 matmul (cannot supply required a_cache for W2 grad)");
    }
    Tensor output = autograd::matmul(intermediates.ffn_gelu_out, W2_, stream,
                                     intermediates.ffn_gelu_out.data,  // cache post_gelu for W2 grad
                                     nullptr);                          // W2 persists
    
    // ISSUE #97 FIX: Add bias b2 with autograd tracking (was bypassing gradient computation)
    if (config_.use_bias && b2_.data) {
        output = autograd::broadcast_add(output, b2_, stream);
    }
    
    // CRITICAL (Issue #56 root cause fix): Return the output tensor!
    // Without this return statement, `output` is destroyed at function end,
    // which triggers destruction of its grad_fn chain, destroying the entire
    // autograd graph DURING the forward pass!
    return output;
}

} // namespace GRIM
