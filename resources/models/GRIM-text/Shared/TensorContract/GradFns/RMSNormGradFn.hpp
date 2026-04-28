#pragma once
//======================================================//
//  RMSNormGradFn.hpp
//  Backward node for RMS normalization.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct RMSNormGradFn : public GradFn {
    bool input_requires_grad = false;
    bool gamma_requires_grad = false;
    float* input_grad = nullptr;
    float* gamma_grad_ptr = nullptr;
    float* gamma_data = nullptr;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_cache;
    std::shared_ptr<float> owned_input_grad;
    const float* cached_input = nullptr;
    std::size_t cached_size = 0;
    int d_model = 0;
    float eps = 1e-5f;

    RMSNormGradFn();
    ~RMSNormGradFn() override;

    void capture_inputs(Tensor& x, Tensor& gamma_tensor, cudaStream_t stream);
    void set_cache_copy(const float* external_cache, std::size_t size, int d, float e, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
