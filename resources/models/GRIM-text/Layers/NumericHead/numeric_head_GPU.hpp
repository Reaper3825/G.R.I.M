//======================================================//
//  Numeric Head Layer - GPU
//  Linear projection from hidden states to numeric prediction
//
//  Output: [total_tokens, 2] — (log_magnitude, sign_logit)
//
//  Forward:  numeric_pred = encoder_output @ W + b
//  Reconstruct: V' = sign(sign_logit) * expm1(clamp(log_magnitude))
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct NumericHeadConfig {
    int d_model = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
};

class NumericHeadLayer {
public:
    static constexpr float kMaxLogMag = 35.0f;  // expm1(35) ~ 1.6e15

    NumericHeadLayer() = delete;

    explicit NumericHeadLayer(const NumericHeadConfig& config,
                              uint64_t seed,
                              cudaStream_t init_stream);

    ~NumericHeadLayer() = default;

    NumericHeadLayer(NumericHeadLayer&& other) noexcept;
    NumericHeadLayer& operator=(NumericHeadLayer&& other) noexcept;

    NumericHeadLayer(const NumericHeadLayer&) = delete;
    NumericHeadLayer& operator=(const NumericHeadLayer&) = delete;

    /// Forward: encoder_output [total_tokens, d_model] -> [total_tokens, 2]
    Tensor forward(Tensor& encoder_output, int total_tokens, cudaStream_t stream);

    /// Reconstruct scalar value from (log_magnitude, sign_logit)
    static float reconstruct(float log_magnitude, float sign_logit);

    // Parameter access for serialization / gradient clipping
    Tensor& weights()  { return weights_; }
    Tensor& bias()     { return bias_; }
    const Tensor& weights() const { return weights_; }
    const Tensor& bias()    const { return bias_; }

    void setStream(cudaStream_t s) { config_.stream = s; }
    void setCublasHandle(cublasHandle_t h) { config_.cublas_handle = h; }

    int d_model() const { return config_.d_model; }

private:
    NumericHeadConfig config_;
    Tensor weights_;  // [2, d_model]  (row-major: output_dim x input_dim)
    Tensor bias_;     // [2]
};

}  // namespace GRIM

#endif  // USE_CUDA
