//======================================================//
//  Feed_Forward_GPU.cu
//  GPU-accelerated FeedForward layer using autograd
//
//  SwiGLU FFN (LLaMA-style fused gate/up projection):
//    gate_up = input @ W_gate_up (+ b_gate_up)   [tokens, 2*d_ff]
//    hidden  = SiLU(gate[:,:d_ff]) * gate[:,d_ff:]  [tokens, d_ff]
//    hidden  = dropout(hidden)
//    output  = hidden @ W_down (+ b_down)         [tokens, d_model]
//
//  Uses fused W_gate_up [d_model, 2*d_ff] to avoid autograd fan-out.
//  The swiglu_fused autograd op splits gate/up internally.
//
//  ISSUE #97 FIX: Bias add now uses autograd::broadcast_add for proper
//  gradient tracking.
//======================================================//

#include "Feed_Forward_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

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
//  FeedForwardLayer Implementation (SwiGLU)
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
    
    // W_gate_up: [d_model, 2*d_ff] fused gate + up projection
    // matmul: input [T, d_model] @ W_gate_up [d_model, 2*d_ff] = [T, 2*d_ff]
    W_gate_up_ = Tensor::zeros({d_model, 2 * d_ff}, stream, "ffn_w_gate_up");
    W_gate_up_.requires_grad_();
    W_gate_up_.ensure_grad();
    Tensor::xavier_uniform_(W_gate_up_, seed, stream);
    
    // W_down: [d_ff, d_model] for matmul: hidden [T, d_ff] @ W_down [d_ff, d_model] = [T, d_model]
    // Issue #142: Scaled by residual_scale after Xavier init (GPT-2 pattern)
    W_down_ = Tensor::zeros({d_ff, d_model}, stream, "ffn_w_down");
    W_down_.requires_grad_();
    W_down_.ensure_grad();
    Tensor::xavier_uniform_(W_down_, seed + 1, stream);
    
    // Apply Issue #142 residual scaling to W_down (down-projection)
    if (residual_scale != 1.0f) {
        const size_t count = W_down_.numel();
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        kernel_ffn_scale_inplace<<<blocks, threads, 0, stream>>>(W_down_.data, count, residual_scale);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("FeedForwardLayer: residual scale kernel failed: " + 
                                     std::string(cudaGetErrorString(err)));
        }
    }
    
    if (config_.use_bias) {
        b_gate_up_ = Tensor::zeros({1, 2 * d_ff}, stream, "ffn_b_gate_up");
        b_gate_up_.requires_grad_();
        b_gate_up_.ensure_grad();
        // Biases initialized to zero (standard practice)
        
        b_down_ = Tensor::zeros({1, d_model}, stream, "ffn_b_down");
        b_down_.requires_grad_();
        b_down_.ensure_grad();
    }
    
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    
    std::fprintf(stderr, "[FeedForwardLayer] SwiGLU self-allocated: W_gate_up=[%d,%d], W_down=[%d,%d], residual_scale=%.6f\n",
                 d_model, 2 * d_ff, d_ff, d_model, residual_scale);
}

FeedForwardLayer::~FeedForwardLayer() {
    // Tensors clean up via RAII
}

FeedForwardLayer::FeedForwardLayer(FeedForwardLayer&& other) noexcept
    : config_(other.config_)
    , W_gate_up_(std::move(other.W_gate_up_))
    , b_gate_up_(std::move(other.b_gate_up_))
    , W_down_(std::move(other.W_down_))
    , b_down_(std::move(other.b_down_)) {
    other.config_.cublas_handle = nullptr;
}

FeedForwardLayer& FeedForwardLayer::operator=(FeedForwardLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        W_gate_up_ = std::move(other.W_gate_up_);
        b_gate_up_ = std::move(other.b_gate_up_);
        W_down_ = std::move(other.W_down_);
        b_down_ = std::move(other.b_down_);
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
//  Forward Pass - SwiGLU with Autograd (Issue #56 Fix)
//======================================================//

Tensor FeedForwardLayer::forward(const Tensor& input, ForwardIntermediates& intermediates,
                                 uint64_t training_step, int layer_idx) {
    // Rule 20: Crash on invalid weights
    if (!W_gate_up_.data || !W_down_.data) {
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
        const auto& s_gu = W_gate_up_.shape.as_2d();
        if (s_gu.rows != config_.d_model || s_gu.cols != 2 * config_.d_ff) {
            throw std::runtime_error("FeedForwardLayer::forward: W_gate_up shape mismatch. Expected [" +
                                     std::to_string(config_.d_model) + "," + std::to_string(2 * config_.d_ff) +
                                     "], got [" + std::to_string(s_gu.rows) + "," + std::to_string(s_gu.cols) + "]");
        }
    }
    {
        const auto& s_d = W_down_.shape.as_2d();
        if (s_d.rows != config_.d_ff || s_d.cols != config_.d_model) {
            throw std::runtime_error("FeedForwardLayer::forward: W_down shape mismatch. Expected [" +
                                     std::to_string(config_.d_ff) + "," + std::to_string(config_.d_model) +
                                     "], got [" + std::to_string(s_d.rows) + "," + std::to_string(s_d.cols) + "]");
        }
    }

    // Set cuBLAS handle for autograd ops
    autograd::set_autograd_cublas_handle(config_.cublas_handle);

    //--------------------------------------------------
    // Step 1: Fused gate + up projection
    // gate_up = input @ W_gate_up   [tokens, d_model] @ [d_model, 2*d_ff] = [tokens, 2*d_ff]
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    intermediates.ffn_gate_up_out = autograd::matmul(input, W_gate_up_, stream,
                                                     input.data,   // cache input for W_gate_up grad
                                                     nullptr);     // W_gate_up persists, no cache needed

    // ISSUE #97 FIX: Add bias with autograd tracking
    if (config_.use_bias && b_gate_up_.data) {
        intermediates.ffn_gate_up_out = autograd::broadcast_add(intermediates.ffn_gate_up_out, b_gate_up_, stream);
    }

    //--------------------------------------------------
    // Step 2: Fused SwiGLU activation
    // hidden = SiLU(gate_up[:, :d_ff]) * gate_up[:, d_ff:]   [tokens, d_ff]
    //--------------------------------------------------
    intermediates.ffn_swiglu_out = autograd::swiglu_fused(intermediates.ffn_gate_up_out, config_.d_ff, stream,
                                                          intermediates.ffn_gate_up_out.data);

    // FFN activation dropout: applied after SwiGLU, before W_down projection
    if (config_.dropout_rate > 0.0f && training_step > 0) {
        const uint64_t ffn_act_dropout_seed = training_step * 2654435761ULL + 300 + layer_idx;
        intermediates.ffn_swiglu_out = autograd::dropout(intermediates.ffn_swiglu_out, config_.dropout_rate,
                                                          ffn_act_dropout_seed, true, stream);
    }

    //--------------------------------------------------
    // Step 3: Down projection
    // output = hidden @ W_down (+ b_down)  [tokens, d_ff] @ [d_ff, d_model] = [tokens, d_model]
    //--------------------------------------------------
    Tensor output = autograd::matmul(intermediates.ffn_swiglu_out, W_down_, stream,
                                     intermediates.ffn_swiglu_out.data,  // cache hidden for W_down grad
                                     nullptr);                            // W_down persists
    
    // ISSUE #97 FIX: Add bias with autograd tracking
    if (config_.use_bias && b_down_.data) {
        output = autograd::broadcast_add(output, b_down_, stream);
    }
    
    // CRITICAL (Issue #56 root cause fix): Return the output tensor!
    return output;
}

} // namespace GRIM
