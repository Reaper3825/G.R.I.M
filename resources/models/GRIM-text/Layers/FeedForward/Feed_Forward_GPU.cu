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

namespace GRIM {

//======================================================//
//  FeedForwardLayer Implementation
//======================================================//

FeedForwardLayer::FeedForwardLayer(const FeedForwardConfig& config,
                                   Tensor& w1, Tensor& b1, Tensor& w2, Tensor& b2)
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

    // Rule 20: Strong shape validation - external weights REQUIRED
    const int d_model = config_.d_model;
    const int d_ff = config_.d_ff;

    // W1: [d_model, d_ff] for matmul: input [T, d_model] @ W1 [d_model, d_ff] = [T, d_ff]
    {
        const auto& s1 = w1.shape.as_2d();
        if (s1.rows != d_model || s1.cols != d_ff) {
            throw std::invalid_argument(
                "FeedForwardLayer: W1 shape mismatch. Expected [" +
                std::to_string(d_model) + "," + std::to_string(d_ff) + "], got [" +
                std::to_string(s1.rows) + "," + std::to_string(s1.cols) + "]"
            );
        }
    }

    // W2: [d_ff, d_model] for matmul: hidden [T, d_ff] @ W2 [d_ff, d_model] = [T, d_model]
    {
        const auto& s2 = w2.shape.as_2d();
        if (s2.rows != d_ff || s2.cols != d_model) {
            throw std::invalid_argument(
                "FeedForwardLayer: W2 shape mismatch. Expected [" +
                std::to_string(d_ff) + "," + std::to_string(d_model) + "], got [" +
                std::to_string(s2.rows) + "," + std::to_string(s2.cols) + "]"
            );
        }
    }

    // Biases optional, but if provided enforce shapes
    if (b1.data) {
        const auto& sb1 = b1.shape.as_2d();
        if (sb1.rows != 1 || sb1.cols != d_ff) {
            throw std::invalid_argument(
                "FeedForwardLayer: b1 shape mismatch. Expected [1," +
                std::to_string(d_ff) + "], got [" +
                std::to_string(sb1.rows) + "," + std::to_string(sb1.cols) + "]"
            );
        }
    }

    if (b2.data) {
        const auto& sb2 = b2.shape.as_2d();
        if (sb2.rows != 1 || sb2.cols != d_model) {
            throw std::invalid_argument(
                "FeedForwardLayer: b2 shape mismatch. Expected [1," +
                std::to_string(d_model) + "], got [" +
                std::to_string(sb2.rows) + "," + std::to_string(sb2.cols) + "]"
            );
        }
    }

    // Create view Tensors referencing external buffers
    // ISSUE #59: Use share_grad() for proper shared_ptr semantics
    W1_ = Tensor::from_ptr(w1.data, w1.shape, false, true, "ffn.W1");
    W1_.share_grad(w1);
    W1_.owns_data = false;

    if (b1.data) {
        b1_ = Tensor::from_ptr(b1.data, b1.shape, false, true, "ffn.b1");
        b1_.share_grad(b1);
        b1_.owns_data = false;
    }

    W2_ = Tensor::from_ptr(w2.data, w2.shape, false, true, "ffn.W2");
    W2_.share_grad(w2);
    W2_.owns_data = false;

    if (b2.data) {
        b2_ = Tensor::from_ptr(b2.data, b2.shape, false, true, "ffn.b2");
        b2_.share_grad(b2);
        b2_.owns_data = false;
    }

    // Set cuBLAS handle for autograd
    autograd::set_autograd_cublas_handle(config_.cublas_handle);

    std::fprintf(stderr, "[FeedForwardLayer] Initialized with external weights: W1=[%zu], W2=[%zu]\n",
                 W1_.numel(), W2_.numel());
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

Tensor FeedForwardLayer::forward(const Tensor& input, ForwardIntermediates& intermediates) {
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

    //--------------------------------------------------
    // Layer 2: output = post_gelu @ W2 + b2
    // The output is also stored in intermediates.ffn_out by EncodingLayer
    //--------------------------------------------------

    // matmul: post_gelu [tokens, d_ff] @ W2 [d_ff, d_model] = output [tokens, d_model]
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
