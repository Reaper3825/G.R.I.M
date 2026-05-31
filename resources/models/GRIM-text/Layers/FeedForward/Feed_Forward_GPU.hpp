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
//  Writes retained FFN tensors directly into ModelForwardOutputs.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <memory>

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

struct FeedForwardParameterViews {
    const Tensor* W_gate = nullptr;
    const Tensor* W1 = nullptr;
    const Tensor* W2 = nullptr;
    const Tensor* b2 = nullptr;
};

//======================================================//
//  FeedForwardLayer - Autograd Implementation
//======================================================//

class FeedForwardLayer {
public:
    // Rule 20: Default constructor deleted - grouped construction HP + registry-owned parameters REQUIRED
    FeedForwardLayer() = delete;
    
    /// Compute-layer constructor.
    /// Durable FFN parameters are owned by ParameterRegistry::StartupParameterRegistry;
    /// callers must pass explicit registry-derived views into forward().
    /// @param hp Grouped FFN construction HP snapshot
    explicit FeedForwardLayer(const HyperParameters::FeedForwardLayerConstructionHP& hp);
    
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
    // Forward Pass - Autograd writing retained FFN tensors into the sink
    //--------------------------------------------------
    /**
     * SwiGLU FFN forward with autograd tracking:
     *   gate   = SiLU(input @ W_gate)
     *   up     = input @ W1
     *   hidden = gate ⊙ up
     *   output = hidden @ W2 + b2
     * 
    * Builds compute graph for automatic backward(). Retained FFN tensors are
    * written directly into the canonical per-call forward sink.
     * 
     * @param input [total_tokens, d_model] - MUST have requires_grad if training
    * @param stream CUDA stream from the caller's forward payload/request
    * @param cublas_handle cuBLAS handle from the caller's forward payload/request
    * @param batch_idx Current batch index for deterministic dropout masks
    * @param dropout_enabled Explicit mode gate for dropout; batch_idx never controls mode
    * @param forward_outputs Canonical per-call forward sink
     * @param layer_idx Encoder layer index (for unique dropout seed per layer)
     */
    void forward(const Tensor& input,
             cudaStream_t stream, cublasHandle_t cublas_handle,
             Forward::ModelForwardOutputs& forward_outputs,
             uint64_t batch_idx, bool dropout_enabled, int layer_idx,
             const FeedForwardParameterViews& parameter_views);

    //--------------------------------------------------
    // NOTE: Backward Pass handled by autograd
    // Call output.backward() after loss computation
    //--------------------------------------------------

private:
    HyperParameters::FeedForwardLayerConstructionHP hp_{};
};

} // namespace GRIM

#endif // USE_CUDA
