//======================================================//
//  numeric_head_GPU.cu
//  GPU-accelerated Numeric Head layer
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: weights [2, d_model], bias [2]
//
//  Forward: numeric_pred = encoder_output @ W^T + b
//  Returns [total_tokens, 2] — (log_magnitude, sign_logit)
//======================================================//

#include "numeric_head_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"  // autograd::broadcast_add

#include <stdexcept>
#include <cstdio>
#include <cmath>

namespace GRIM {

//======================================================//
//  Constructor
//======================================================//
NumericHeadLayer::NumericHeadLayer(const NumericHeadConfig& config,
                                   uint64_t seed,
                                   cudaStream_t init_stream)
    : config_(config)
{
    if (config_.d_model <= 0) {
        throw std::runtime_error("NumericHeadLayer: d_model must be positive");
    }
    if (!init_stream) {
        throw std::runtime_error("NumericHeadLayer: init_stream is NULL");
    }

    // Weights: [2, d_model] — row-major so matmul is output = input @ W^T
    weights_ = Tensor::zeros({2, config_.d_model}, init_stream, "numeric_head.weights");
    weights_.requires_grad_();
    weights_.ensure_grad();
    Tensor::xavier_uniform_(weights_, seed, init_stream);

    // Bias: [2] — initialized to zero
    bias_ = Tensor::zeros({2}, init_stream, "numeric_head.bias");
    bias_.requires_grad_();
    bias_.ensure_grad();

    fprintf(stdout, "[NumericHeadLayer] Initialized: d_model=%d, weights=[2, %d], Xavier seed=%llu\n",
            config_.d_model, config_.d_model, (unsigned long long)seed);
}

//======================================================//
//  Move Operations
//======================================================//
NumericHeadLayer::NumericHeadLayer(NumericHeadLayer&& other) noexcept
    : config_(other.config_)
    , weights_(std::move(other.weights_))
    , bias_(std::move(other.bias_))
{
    other.config_.cublas_handle = nullptr;
}

NumericHeadLayer& NumericHeadLayer::operator=(NumericHeadLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        weights_ = std::move(other.weights_);
        bias_ = std::move(other.bias_);
        other.config_.cublas_handle = nullptr;
    }
    return *this;
}

//======================================================//
//  Forward: encoder_output @ W^T + b
//======================================================//
Tensor NumericHeadLayer::forward(Tensor& encoder_output, int total_tokens, cudaStream_t stream) {
    if (!config_.cublas_handle) {
        throw std::runtime_error("NumericHeadLayer::forward: cublas_handle is NULL");
    }
    if (!encoder_output.data) {
        throw std::runtime_error("NumericHeadLayer::forward: encoder_output has null data");
    }

    weights_.requires_grad = true;

    const float* a_cache = encoder_output.data;

    Tensor output = autograd::matmul(
        encoder_output,
        weights_,
        stream,
        a_cache,
        nullptr,
        true  // transpose_b=true: output = input @ W^T  -> [total_tokens, 2]
    );

    // Add bias via autograd so grad_bias flows (Issue: B3 - kernelNumericHeadBias bypassed autograd)
    if (bias_.data) {
        output = autograd::broadcast_add(output, bias_, stream);
    }

    return output;
}

//======================================================//
//  Reconstruct scalar value from head outputs
//======================================================//
float NumericHeadLayer::reconstruct(float log_magnitude, float sign_logit) {
    if (!std::isfinite(log_magnitude) || !std::isfinite(sign_logit)) return 0.0f;

    float clamped = std::fmin(std::fmax(log_magnitude, 0.0f), kMaxLogMag);
    float sign = (sign_logit >= 0.0f) ? 1.0f : -1.0f;
    float mag = std::expm1(clamped);
    return sign * mag;
}

}  // namespace GRIM
