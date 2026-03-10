//======================================================//
//  Feed_Forward_GPU.hpp
//  GPU-accelerated FeedForward layer using autograd
//  
//  SwiGLU gated MLP:
//    gate  = SiLU(input @ W_gate)
//    up    = input @ W1
//    hidden = gate ⊙ up
//    output = hidden @ W2 + b2
//  
//  Uses autograd::matmul, autograd::silu, autograd::elementwise_mul
//  for automatic differentiation.
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
    bool use_bias = true;        // When false, skip bias addition (b2)
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
    
    /// Self-allocating constructor (Pattern B: layer self-management)
    /// Allocates and Xavier-initializes W_gate, W1, W2, b2 on GPU.
    /// Layer OWNS the memory (owns_data=true). Registers with autograd via ensure_grad().
    /// @param config Layer configuration (d_model, d_ff, cublas_handle, stream REQUIRED)
    /// @param seed   Xavier initialization seed
    /// @param residual_scale Issue #142: Scale W2 by 1/sqrt(2*num_layers) after Xavier init
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
    
    // Tensor weight accessors (for training/serialization/buildParameterGroups)
    Tensor& W1() { return W1_; }
    Tensor& W_gate() { return W_gate_; }
    Tensor& W2() { return W2_; }
    Tensor& b2() { return b2_; }
    const Tensor& W1() const { return W1_; }
    const Tensor& W_gate() const { return W_gate_; }
    const Tensor& W2() const { return W2_; }
    const Tensor& b2() const { return b2_; }

    //--------------------------------------------------
    // Forward Pass - Autograd with ForwardIntermediates (Issue #56 Fix)
    //--------------------------------------------------
    /**
     * SwiGLU FFN forward with autograd tracking:
     *   gate   = SiLU(input @ W_gate)
     *   up     = input @ W1
     *   hidden = gate ⊙ up
     *   output = hidden @ W2 + b2
     * 
     * Builds compute graph for automatic backward().
     * 
     * CRITICAL: Intermediate tensors are stored in ForwardIntermediates
     * to keep the autograd graph alive. Without this, grad_fn objects
     * are destroyed when forward() returns, causing use-after-free
     * in backward pass.
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

    // Weight Tensors with autograd (requires_grad=true)
    // SwiGLU uses three projections: gate, up, down
    Tensor W_gate_; // [d_model, d_ff] - gate projection (SiLU applied)
    Tensor W1_;     // [d_model, d_ff] - up projection (linear)
    Tensor W2_;     // [d_ff, d_model] - down projection
    Tensor b2_;     // [d_model]       - down projection bias
};

} // namespace GRIM

#endif // USE_CUDA
