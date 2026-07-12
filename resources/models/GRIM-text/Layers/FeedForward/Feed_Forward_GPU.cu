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
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
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

namespace {

void validateFeedForwardHP(const HyperParameters::FeedForwardLayerConstructionHP& hp,
                           const char* context) {
    const float residual_projection_init_gain = hp.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error(std::string(context) + ": residual_projection_init_gain must be a positive finite value from feedForwardLayerConstructionHP");
    }
    if (hp.d_model <= 0) {
        throw std::runtime_error(std::string(context) + ": d_model must be > 0, got " + std::to_string(hp.d_model));
    }
    if (hp.d_ff <= 0) {
        throw std::runtime_error(std::string(context) + ": d_ff must be > 0, got " + std::to_string(hp.d_ff));
    }
    if (!std::isfinite(hp.dropout_rate) || hp.dropout_rate < 0.0f || hp.dropout_rate >= 1.0f) {
        throw std::runtime_error(std::string(context) + ": dropout_rate must be finite and in [0,1), got " +
                                 std::to_string(hp.dropout_rate));
    }
}

void validateFeedForwardParameters(const FeedForwardParameterTensors& parameters,
                                   const HyperParameters::FeedForwardLayerConstructionHP& hp,
                                   const char* context) {
    if (!parameters.W_gate.data || !parameters.W1.data || !parameters.W2.data) {
        throw std::runtime_error(std::string(context) + ": W_gate/W1/W2 parameter tensors must have allocated data");
    }
    if (hp.output_bias_enabled) {
        if (!parameters.b2.data) {
            throw std::runtime_error(std::string(context) + ": output_bias_enabled=true requires allocated b2 parameter tensor");
        }
    } else if (parameters.b2.data) {
        throw std::runtime_error(std::string(context) + ": output_bias_enabled=false but b2 parameter tensor is allocated");
    }
}

} // namespace

//--------------------------------------------------
// Compute-layer constructor. Durable tensors live in ParameterRegistry.
//--------------------------------------------------
FeedForwardLayer::FeedForwardLayer(const HyperParameters::FeedForwardLayerConstructionHP& hp)
    : hp_(hp) {
    validateFeedForwardHP(hp_, "FeedForwardLayer::FeedForwardLayer");
}

FeedForwardLayer::~FeedForwardLayer() {
    // No parameter ownership.
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : hp_(other.hp_) {
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        hp_ = other.hp_;
    }
    return *this;
}

//======================================================//
//  Forward Pass - SwiGLU writing retained tensors into the shared sink
//======================================================//

void FeedForwardLayer::forward(const Tensor& input,
                               cudaStream_t stream,
                               cublasHandle_t cublas_handle,
                               Forward::ModelForwardOutputs& forward_outputs,
                               int layer_idx,
                               const FeedForwardParameterTensors& parameter_tensors) {
    if (layer_idx < 0) {
        throw std::runtime_error("FeedForwardLayer::forward: layer_idx must be >= 0, got " + std::to_string(layer_idx));
    }
    const size_t layer_slot = static_cast<size_t>(layer_idx);
    forward_outputs.validateLayerIndex(layer_slot, "FeedForwardLayer::forward");
    Tensor& ffn_gate_out = forward_outputs.ffn_gate_out_per_layer[layer_slot];
    Tensor& ffn_silu_out = forward_outputs.ffn_silu_out_per_layer[layer_slot];
    Tensor& ffn_linear1_out = forward_outputs.ffn_linear1_out_per_layer[layer_slot];
    Tensor& ffn_swiglu_out = forward_outputs.ffn_swiglu_out_per_layer[layer_slot];
    Tensor& ffn_out = forward_outputs.ffn_out_per_layer[layer_slot];
    validateFeedForwardParameters(parameter_tensors, hp_, "FeedForwardLayer::forward");
    const Tensor& W_gate = parameter_tensors.W_gate;
    const Tensor& W1 = parameter_tensors.W1;
    const Tensor& W2 = parameter_tensors.W2;
    const Tensor* b2 = hp_.output_bias_enabled ? &parameter_tensors.b2 : nullptr;

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
    ffn_gate_out = autograd::matmul(input, W_gate, stream);
    ffn_silu_out = autograd::silu(ffn_gate_out, stream,
                                  ffn_gate_out.data);

    //--------------------------------------------------
    // SwiGLU Up Path: up = input @ W1
    //--------------------------------------------------
    ffn_linear1_out = autograd::matmul(input, W1, stream);

    //--------------------------------------------------
    // SwiGLU Combine: hidden = gate ⊙ up
    //--------------------------------------------------
    ffn_swiglu_out = autograd::elementwise_mul(
        ffn_silu_out, ffn_linear1_out, stream);

    //--------------------------------------------------
    // Down Projection: output = hidden @ W2 + b2
    //--------------------------------------------------
    if (!ffn_swiglu_out.data) {
        throw std::runtime_error("FeedForwardLayer::forward: ffn_swiglu_out.data is NULL before W2 matmul");
    }
    ffn_out = autograd::matmul(ffn_swiglu_out, W2, stream);
    
    if (hp_.output_bias_enabled) {
        ffn_out = autograd::broadcast_add(ffn_out, *b2, stream);
    }
}

} // namespace GRIM
