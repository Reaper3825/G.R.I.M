//======================================================//
//  Feed_Forward_GPU.cu
//  GPU-accelerated FeedForward layer using autograd
//
//  SwiGLU gated MLP:
//    gate   = SiLU(input @ W_gate)
//    up     = input @ W1
//    hidden = gate ⊙ up
//    output = hidden @ W2 + b2
//
//  ISSUE #97 FIX: Bias add now uses autograd::broadcast_add for proper
//  gradient tracking. Previously used raw kernels which bypassed autograd,
//  causing b2 to receive ZERO gradients (frozen bias).
//======================================================//

#include "Feed_Forward_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"  // Issue #97: autograd::broadcast_add

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace GRIM {

//======================================================//
//  FeedForwardLayer Implementation
//======================================================//

//--------------------------------------------------
// Self-allocating constructor (Pattern B: layer self-management)
// Follows ScratchBlockLayer pattern: allocate → ensure_grad → Xavier init
//--------------------------------------------------
FeedForwardLayer::FeedForwardLayer(const HyperParameters::FeedForwardLayerConstructionHP& hp, uint64_t seed,
                                   cudaStream_t init_stream)
    : hp_(hp) {
    if (!init_stream) {
        throw std::runtime_error("FeedForwardLayer: init_stream is NULL");
    }
    const float residual_projection_init_gain = hp_.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("FeedForwardLayer: residual_projection_init_gain must be a positive finite value from feedForwardLayerConstructionHP");
    }
    if (hp_.d_model <= 0) {
        throw std::runtime_error("FeedForwardLayer: d_model must be > 0, got " + std::to_string(hp_.d_model));
    }
    if (hp_.d_ff <= 0) {
        throw std::runtime_error("FeedForwardLayer: d_ff must be > 0, got " + std::to_string(hp_.d_ff));
    }
    if (!std::isfinite(hp_.dropout_rate) || hp_.dropout_rate < 0.0f || hp_.dropout_rate >= 1.0f) {
        throw std::runtime_error("FeedForwardLayer: dropout_rate must be finite and in [0,1), got " +
                                 std::to_string(hp_.dropout_rate));
    }
    
    const int d_model = hp_.d_model;
    const int d_ff = hp_.d_ff;
    const cudaStream_t stream = init_stream;
    
    // W_gate: [d_model, d_ff] gate projection (SiLU applied to this path)
    W_gate_ = Tensor::zeros({d_model, d_ff}, stream, "ffn_w_gate");
    W_gate_.requires_grad_();
    W_gate_.ensure_grad();
    Tensor::xavier_uniform_(W_gate_, seed, stream);
    
    // W1: [d_model, d_ff] up projection (linear, no activation)
    W1_ = Tensor::zeros({d_model, d_ff}, stream, "ffn_w1");
    W1_.requires_grad_();
    W1_.ensure_grad();
    Tensor::xavier_uniform_(W1_, seed + 1, stream);
    
    // W2: [d_ff, d_model] down projection
    // Residual projection startup init: Xavier with explicit depth gain
    W2_ = Tensor::zeros({d_ff, d_model}, stream, "ffn_w2");
    W2_.requires_grad_();
    W2_.ensure_grad();
    Tensor::xavier_uniform_with_gain_(W2_, seed + 2, residual_projection_init_gain, stream);
    
    if (hp_.use_bias) {
        b2_ = Tensor::zeros({1, d_model}, stream, "ffn_b2");
        b2_.requires_grad_();
        b2_.ensure_grad();
    }
    
    std::fprintf(stderr, "[FeedForwardLayer] SwiGLU self-allocated weights: W_gate=[%d,%d], W1=[%d,%d], W2=[%d,%d], residual_projection_init_gain=%.6f\n",
                 d_model, d_ff, d_model, d_ff, d_ff, d_model, residual_projection_init_gain);
}

FeedForwardLayer::~FeedForwardLayer() {
    // Tensors clean up via RAII
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : hp_(other.hp_)
    , W_gate_(std::move(other.W_gate_))
    , W1_(std::move(other.W1_))
    , W2_(std::move(other.W2_))
    , b2_(std::move(other.b2_)) {
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        hp_ = other.hp_;
        W_gate_ = std::move(other.W_gate_);
        W1_ = std::move(other.W1_);
        W2_ = std::move(other.W2_);
        b2_ = std::move(other.b2_);
    }
    return *this;
}

//======================================================//
//  Forward Pass - SwiGLU with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor FeedForwardLayer::forward(const Tensor& input, ForwardIntermediates& intermediates,
                                 cudaStream_t stream, cublasHandle_t cublas_handle,
                                 uint64_t batch_idx, bool dropout_enabled, int layer_idx,
                                 const FeedForwardParameterViews* parameter_views) {
    const Tensor& W_gate = (parameter_views && parameter_views->W_gate) ? *parameter_views->W_gate : W_gate_;
    const Tensor& W1 = (parameter_views && parameter_views->W1) ? *parameter_views->W1 : W1_;
    const Tensor& W2 = (parameter_views && parameter_views->W2) ? *parameter_views->W2 : W2_;
    const Tensor& b2 = (parameter_views && parameter_views->b2) ? *parameter_views->b2 : b2_;

    // Rule 20: Crash on invalid weights
    if (!W_gate.data || !W1.data || !W2.data) {
        throw std::runtime_error("FeedForwardLayer::forward: selected weights not set (W_gate/W1/W2)");
    }
    if (!stream) {
        throw std::runtime_error("FeedForwardLayer::forward: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error("FeedForwardLayer::forward: cublas_handle is NULL");
    }

    // Set cuBLAS handle for autograd ops
    autograd::set_autograd_cublas_handle(cublas_handle);

    if (!input.data) {
        throw std::runtime_error("FeedForwardLayer::forward: input.data is NULL");
    }

    //--------------------------------------------------
    // SwiGLU Gate Path: gate = SiLU(input @ W_gate)
    //--------------------------------------------------
    intermediates.ffn_gate_out = autograd::matmul(input, W_gate, stream,
                                                   input.data, nullptr);
    intermediates.ffn_silu_out = autograd::silu(intermediates.ffn_gate_out, stream,
                                                intermediates.ffn_gate_out.data);

    //--------------------------------------------------
    // SwiGLU Up Path: up = input @ W1
    //--------------------------------------------------
    intermediates.ffn_linear1_out = autograd::matmul(input, W1, stream,
                                                     input.data, nullptr);

    //--------------------------------------------------
    // SwiGLU Combine: hidden = gate ⊙ up
    //--------------------------------------------------
    intermediates.ffn_swiglu_out = autograd::elementwise_mul(
        intermediates.ffn_silu_out, intermediates.ffn_linear1_out, stream);

    // Activation dropout: applied after SwiGLU gating, before W2 projection
    if (hp_.dropout_rate > 0.0f && dropout_enabled) {
        const uint64_t ffn_act_dropout_seed = batch_idx * 2654435761ULL + 300 + layer_idx;
        const uint64_t ffn_act_dropout_mask_stream = 0x0003000000000000ULL + static_cast<uint64_t>(layer_idx);
        intermediates.ffn_swiglu_out = autograd::dropout(intermediates.ffn_swiglu_out, hp_.dropout_rate,
                                                          ffn_act_dropout_seed, stream,
                                                          ffn_act_dropout_mask_stream);
    }

    //--------------------------------------------------
    // Down Projection: output = hidden @ W2 + b2
    //--------------------------------------------------
    if (!intermediates.ffn_swiglu_out.data) {
        throw std::runtime_error("FeedForwardLayer::forward: ffn_swiglu_out.data is NULL before W2 matmul");
    }
    Tensor output = autograd::matmul(intermediates.ffn_swiglu_out, W2, stream,
                                     intermediates.ffn_swiglu_out.data, nullptr);
    
    if (hp_.use_bias && b2.data) {
        output = autograd::broadcast_add(output, b2, stream);
    }
    
    return output;
}

} // namespace GRIM
