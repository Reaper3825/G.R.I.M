#pragma once
//======================================================//
//  SigmoidGradFn.hpp
//  Backward node for elementwise sigmoid: y = sigmoid(x).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct SigmoidGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_input = nullptr;
    std::size_t cached_size = 0;

    SigmoidGradFn();
    ~SigmoidGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void set_cache_ref(const float* data, std::size_t size);
    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
