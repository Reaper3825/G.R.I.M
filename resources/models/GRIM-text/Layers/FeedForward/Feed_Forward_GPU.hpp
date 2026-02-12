//======================================================//
//  Feed_Forward_GPU.hpp
//  GPU-accelerated FeedForward layer using autograd
//  
//  Two-layer MLP: Linear -> GELU -> Linear
//  Uses autograd::matmul and autograd::gelu for automatic differentiation
//  
//  ISSUE #56 FIX: Forward now accepts ForwardIntermediates& to keep
//  intermediate tensors alive until backward completes.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <memory>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TensorContract/ForwardIntermediates.hpp"

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//

struct FeedForwardConfig {
    int d_model = 0;           // Input/output dimension debug was 768
    int d_ff = 0;             // Hidden (intermediate) dimension debug was 3072
    float dropout_rate = 0.1f;   // Dropout probability
    bool use_bias = true;        // When false, skip bias addition (b1, b2)
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;  // Rule 22: MUST be training_state.cublas_handle
};

//======================================================//
//  FeedForwardLayer - Autograd Implementation
//======================================================//

class FeedForwardLayer {
public:
    // Rule 20: Default constructor deleted - config with valid cublas_handle REQUIRED
    FeedForwardLayer() = delete;
    // Constructor takes external weights (required) - no allocation path
    explicit FeedForwardLayer(const FeedForwardConfig& config, 
                             Tensor& w1, Tensor& b1, Tensor& w2, Tensor& b2);
    ~FeedForwardLayer();

    // Prevent copy (cuBLAS handle, Tensor ownership)
    FeedForwardLayer(const FeedForwardLayer&) = delete;
    FeedForwardLayer& operator=(const FeedForwardLayer&) = delete;

    // Allow move
    FeedForwardLayer(FeedForwardLayer&& other) noexcept;
    FeedForwardLayer& operator=(FeedForwardLayer&& other) noexcept;

    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------
    void setConfig(const FeedForwardConfig& cfg);
    const FeedForwardConfig& config() const noexcept { return config_; }

    //--------------------------------------------------
    // Weight Management (external weights only)
    //--------------------------------------------------
    // Weights are set via constructor - no separate allocation/configuration methods
    
    // Tensor weight accessors (for training/serialization)
    Tensor& W1() { return W1_; }
    Tensor& b1() { return b1_; }
    Tensor& W2() { return W2_; }
    Tensor& b2() { return b2_; }
    const Tensor& W1() const { return W1_; }
    const Tensor& b1() const { return b1_; }
    const Tensor& W2() const { return W2_; }
    const Tensor& b2() const { return b2_; }

    //--------------------------------------------------
    // Forward Pass - Autograd with ForwardIntermediates (Issue #56 Fix)
    //--------------------------------------------------
    /**
     * FFN forward with autograd tracking:
     *   hidden = GELU(input @ W1^T + b1)
     *   output = hidden @ W2^T + b2
     * 
     * Builds compute graph for automatic backward().
     * 
     * CRITICAL: Intermediate tensors (pre_gelu, post_gelu) are stored in
     * ForwardIntermediates to keep the autograd graph alive. Without this,
     * grad_fn objects are destroyed when forward() returns, causing 
     * use-after-free in backward pass.
     * 
     * @param input [total_tokens, d_model] - MUST have requires_grad if training
     * @param intermediates Storage for intermediate tensors (REQUIRED for autograd)
     * @return output [total_tokens, d_model] with grad_fn attached
     */
    Tensor forward(const Tensor& input, ForwardIntermediates& intermediates);

    //--------------------------------------------------
    // NOTE: Backward Pass handled by autograd
    // Call output.backward() after loss computation
    //--------------------------------------------------

private:
    FeedForwardConfig config_{};

    // Weight Tensors with autograd (requires_grad=true)
    Tensor W1_;    // [d_ff, d_model]
    Tensor b1_;    // [d_ff]
    Tensor W2_;    // [d_model, d_ff]
    Tensor b2_;    // [d_model]
};

} // namespace GRIM

#endif // USE_CUDA
