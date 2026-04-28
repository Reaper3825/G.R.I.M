#pragma once
//======================================================//
//  MulScalarGradFn.hpp
//  Backward node for element-wise multiply by constant: y = x * c.
//
//  Owns:
//    - struct MulScalarGradFn          (declared here)
//    - kernel_mul_scalar_forward       (defined in MulScalarGradFn.cu)
//    - kernel_mul_scalar_backward      (defined in MulScalarGradFn.cu)
//    - autograd::mul_scalar(...)       (forward op; defined in MulScalarGradFn.cu)
//
//  Backward: grad_x += grad_y * scalar (no saved tensor needed).
//======================================================//

#include "TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct MulScalarGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float scalar = 0.0f;
    std::size_t count = 0;

    MulScalarGradFn();
    ~MulScalarGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
