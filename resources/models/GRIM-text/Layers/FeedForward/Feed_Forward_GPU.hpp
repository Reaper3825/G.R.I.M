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
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

//======================================================//
//  FeedForwardLayer - Autograd Implementation
//======================================================//

class FeedForwardLayer {
public:
    // Rule 20: Default constructor deleted - grouped construction HP + init stream REQUIRED
    FeedForwardLayer() = delete;
    
    /// Self-allocating constructor (Pattern B: layer self-management)
    /// Allocates and Xavier-initializes W_gate, W1, W2, b2 on GPU.
    /// Layer OWNS the memory (owns_data=true). Registers with autograd via ensure_grad().
    /// @param hp     Grouped FFN construction HP snapshot, including residual_projection_init_gain
    /// @param seed   Xavier initialization seed
    /// @param init_stream CUDA stream for self-allocation during startup/model assembly
    explicit FeedForwardLayer(const HyperParameters::FeedForwardLayerConstructionHP& hp, uint64_t seed,
                              cudaStream_t init_stream);
    
    ~FeedForwardLayer();

    // Prevent copy (cuBLAS handle, Tensor ownership)
    FeedForwardLayer(const FeedForwardLayer&) = delete;
    FeedForwardLayer& operator=(const FeedForwardLayer&) = delete;

    // Allow move
    FeedForwardLayer(FeedForwardLayer&& other) noexcept;
    FeedForwardLayer& operator=(FeedForwardLayer&& other) noexcept;

    //--------------------------------------------------
    // Grouped HP snapshot
    //--------------------------------------------------
    const HyperParameters::FeedForwardLayerConstructionHP& hp() const noexcept { return hp_; }

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
    * @param stream CUDA stream from the caller's forward payload/request
    * @param cublas_handle cuBLAS handle from the caller's forward payload/request
    * @param batch_idx Current batch index for deterministic dropout masks
    * @param dropout_enabled Explicit mode gate for dropout; batch_idx never controls mode
     * @param layer_idx Encoder layer index (for unique dropout seed per layer)
     * @return output [total_tokens, d_model] with grad_fn attached
     */
    Tensor forward(const Tensor& input, ForwardIntermediates& intermediates,
                cudaStream_t stream, cublasHandle_t cublas_handle,
                uint64_t batch_idx = 0, bool dropout_enabled = false, int layer_idx = 0);

    //--------------------------------------------------
    // NOTE: Backward Pass handled by autograd
    // Call output.backward() after loss computation
    //--------------------------------------------------

private:
    HyperParameters::FeedForwardLayerConstructionHP hp_{};

    // Weight Tensors with autograd (requires_grad=true)
    // SwiGLU uses three projections: gate, up, down
    Tensor W_gate_; // [d_model, d_ff] - gate projection (SiLU applied)
    Tensor W1_;     // [d_model, d_ff] - up projection (linear)
    Tensor W2_;     // [d_ff, d_model] - down projection
    Tensor b2_;     // [d_model]       - down projection bias
};

} // namespace GRIM

#endif // USE_CUDA
