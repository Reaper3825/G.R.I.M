//======================================================//
//  Feed_Forward_GPU.hpp
//  GPU-accelerated FeedForward layer using autograd
//  
//  SwiGLU FFN (LLaMA-style fused gate/up projection):
//    gate_up = input @ W_gate_up  [tokens, 2*d_ff]
//    hidden  = SiLU(gate_up[:, :d_ff]) * gate_up[:, d_ff:]
//    output  = hidden @ W_down + b_down
//
//  Uses fused W_gate_up [d_model, 2*d_ff] to avoid autograd fan-out.
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
    int d_model = 0;           // Input/output dimension
    int d_ff = 0;             // Hidden (intermediate) dimension
    float dropout_rate = 0.1f;   // Dropout probability
    bool use_bias = true;        // When false, skip bias addition (b_gate_up, b_down)
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;  // Rule 22: MUST be training_state.cublas_handle
};

//======================================================//
//  FeedForwardLayer - SwiGLU Autograd Implementation
//======================================================//

class FeedForwardLayer {
public:
    // Rule 20: Default constructor deleted - config with valid cublas_handle REQUIRED
    FeedForwardLayer() = delete;
    
    /// Self-allocating constructor (Pattern B: layer self-management)
    /// Allocates and Xavier-initializes W_gate_up, W_down, biases on GPU.
    /// Layer OWNS the memory (owns_data=true). Registers with autograd via ensure_grad().
    /// @param config Layer configuration (d_model, d_ff, cublas_handle, stream REQUIRED)
    /// @param seed   Xavier initialization seed
    /// @param residual_scale Issue #142: Scale W_down by 1/sqrt(2*num_layers) after Xavier init
    explicit FeedForwardLayer(const FeedForwardConfig& config, uint64_t seed, float residual_scale = 1.0f);
    
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
    // Weight Management (Pattern B: self-allocated)
    //--------------------------------------------------
    
    // SwiGLU weight accessors (for training/serialization/buildParameterGroups)
    Tensor& W_gate_up() { return W_gate_up_; }
    Tensor& b_gate_up() { return b_gate_up_; }
    Tensor& W_down() { return W_down_; }
    Tensor& b_down() { return b_down_; }
    const Tensor& W_gate_up() const { return W_gate_up_; }
    const Tensor& b_gate_up() const { return b_gate_up_; }
    const Tensor& W_down() const { return W_down_; }
    const Tensor& b_down() const { return b_down_; }

    //--------------------------------------------------
    // Forward Pass - Autograd with ForwardIntermediates (Issue #56 Fix)
    //--------------------------------------------------
    /**
     * SwiGLU FFN forward with autograd tracking:
     *   gate_up = input @ W_gate_up (+ b_gate_up)    [tokens, 2*d_ff]
     *   hidden  = SiLU(gate_up[:,:d_ff]) * gate_up[:,d_ff:]   [tokens, d_ff]
     *   hidden  = dropout(hidden)
     *   output  = hidden @ W_down (+ b_down)         [tokens, d_model]
     * 
     * Builds compute graph for automatic backward().
     * 
     * CRITICAL: Intermediate tensors are stored in ForwardIntermediates
     * to keep the autograd graph alive. Without this, grad_fn objects are
     * destroyed when forward() returns, causing use-after-free in backward.
     * 
     * @param input [total_tokens, d_model] - MUST have requires_grad if training
     * @param intermediates Storage for intermediate tensors (REQUIRED for autograd)
     * @param training_step Current training step (0 = inference, no dropout)
     * @param layer_idx Encoder layer index (for unique dropout seed per layer)
     * @return output [total_tokens, d_model] with grad_fn attached
     */
    Tensor forward(const Tensor& input, ForwardIntermediates& intermediates,
                   uint64_t training_step = 0, int layer_idx = 0);

    //--------------------------------------------------
    // NOTE: Backward Pass handled by autograd
    // Call output.backward() after loss computation
    //--------------------------------------------------

private:
    FeedForwardConfig config_{};

    // SwiGLU Weight Tensors with autograd (requires_grad=true)
    Tensor W_gate_up_;  // [d_model, 2*d_ff] fused gate + up projection
    Tensor b_gate_up_;  // [2*d_ff] fused bias (optional, controlled by use_bias)
    Tensor W_down_;     // [d_ff, d_model] down projection
    Tensor b_down_;     // [d_model] down bias (optional, controlled by use_bias)
};

} // namespace GRIM

#endif // USE_CUDA
