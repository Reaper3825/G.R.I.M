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
    // Issue #142: Scaled by residual_scale after Xavier init (GPT-2 pattern)
    W2_ = Tensor::zeros({d_ff, d_model}, stream, "ffn_w2");
    W2_.requires_grad_();
    W2_.ensure_grad();
    Tensor::xavier_uniform_(W2_, seed + 2, stream);
    
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
        b2_ = Tensor::zeros({1, d_model}, stream, "ffn_b2");
        b2_.requires_grad_();
        b2_.ensure_grad();
    }
    
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    
    std::fprintf(stderr, "[FeedForwardLayer] SwiGLU self-allocated weights: W_gate=[%d,%d], W1=[%d,%d], W2=[%d,%d], residual_scale=%.6f\n",
                 d_model, d_ff, d_model, d_ff, d_ff, d_model, residual_scale);
}

FeedForwardLayer::~FeedForwardLayer() {
    // Tensors clean up via RAII
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : config_(other.config_)
    , W_gate_(std::move(other.W_gate_))
    , W1_(std::move(other.W1_))
    , W2_(std::move(other.W2_))
    , b2_(std::move(other.b2_)) {
    other.config_.cublas_handle = nullptr;
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        W_gate_ = std::move(other.W_gate_);
        W1_ = std::move(other.W1_);
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
//  Forward Pass - SwiGLU with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor FeedForwardLayer::forward(const Tensor& input, ForwardIntermediates& intermediates,
                                 uint64_t training_step, int layer_idx) {
    // Rule 20: Crash on invalid weights
    if (!W_gate_.data || !W1_.data || !W2_.data) {
        throw std::runtime_error("FeedForwardLayer::forward: weights not set (W_gate/W1/W2)");
    }
    if (!config_.stream) {
        throw std::runtime_error("FeedForwardLayer::forward: stream is NULL");
    }

    const cudaStream_t stream = config_.stream;

    // Extract dimensions from input shape
    const auto& in_shape = input.shape.as_2d();
    const int d_model = in_shape.cols;

    if (d_model != config_.d_model) {
        throw std::runtime_error("FeedForwardLayer::forward: input d_model mismatch (" +
                                 std::to_string(d_model) + " vs " + std::to_string(config_.d_model) + ")");
    }

    // Sanity-check weight shapes
    {
        const auto& sg = W_gate_.shape.as_2d();
        if (sg.rows != config_.d_model || sg.cols != config_.d_ff) {
            throw std::runtime_error("FeedForwardLayer::forward: W_gate shape mismatch. Expected [" +
                                     std::to_string(config_.d_model) + "," + std::to_string(config_.d_ff) +
                                     "], got [" + std::to_string(sg.rows) + "," + std::to_string(sg.cols) + "]");
        }
    }
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

    if (!input.data) {
        throw std::runtime_error("FeedForwardLayer::forward: input.data is NULL");
    }

    //--------------------------------------------------
    // SwiGLU Gate Path: gate = SiLU(input @ W_gate)
    //--------------------------------------------------
    intermediates.ffn_gate_out = autograd::matmul(input, W_gate_, stream,
                                                   input.data, nullptr);
    intermediates.ffn_silu_out = autograd::silu(intermediates.ffn_gate_out, stream,
                                                intermediates.ffn_gate_out.data);

    //--------------------------------------------------
    // SwiGLU Up Path: up = input @ W1
    //--------------------------------------------------
    intermediates.ffn_linear1_out = autograd::matmul(input, W1_, stream,
                                                     input.data, nullptr);

    //--------------------------------------------------
    // SwiGLU Combine: hidden = gate ⊙ up
    //--------------------------------------------------
    intermediates.ffn_swiglu_out = autograd::elementwise_mul(
        intermediates.ffn_silu_out, intermediates.ffn_linear1_out, stream);

    // Activation dropout: applied after SwiGLU gating, before W2 projection
    if (config_.dropout_rate > 0.0f && training_step > 0) {
        const uint64_t ffn_act_dropout_seed = training_step * 2654435761ULL + 300 + layer_idx;
        intermediates.ffn_swiglu_out = autograd::dropout(intermediates.ffn_swiglu_out, config_.dropout_rate,
                                                          ffn_act_dropout_seed, true, stream);
    }

    //--------------------------------------------------
    // Down Projection: output = hidden @ W2 + b2
    //--------------------------------------------------
    if (!intermediates.ffn_swiglu_out.data) {
        throw std::runtime_error("FeedForwardLayer::forward: ffn_swiglu_out.data is NULL before W2 matmul");
    }
    Tensor output = autograd::matmul(intermediates.ffn_swiglu_out, W2_, stream,
                                     intermediates.ffn_swiglu_out.data, nullptr);
    
    if (config_.use_bias && b2_.data) {
        output = autograd::broadcast_add(output, b2_, stream);
    }
    
    return output;
}

} // namespace GRIM
