#pragma once
//======================================================//
//  ResidualAddGradFn.hpp
//  Specialized backward node for residual / skip connections.
//  Forward:  y = x + residual
//  Backward: grad_x = grad_y, grad_residual = grad_y (both pass-through)
//
//  Functionally equivalent to AddGradFn for the gradient math, but kept
//  as its own op so that "residual_add" shows up distinctly in op_name
//  traces / logs. The forward delegates to TensorContract::add().
//
//  Owns:
//    - struct ResidualAddGradFn      (declared here)
//    - autograd::residual_add(...)   (forward op; defined in .cu)
//
//  Backward uses TensorContract/GradientAccumulation.hpp; do not add
//  per-TU kernel_accumulate_grad copies here.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ResidualAddGradFn : public GradFn {
    bool input_requires_grad = false;
    bool residual_requires_grad = false;
    float* input_grad = nullptr;
    float* residual_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    std::shared_ptr<float> owned_residual_grad;
    TensorContract::TensorShape input_shape;
    TensorContract::TensorShape residual_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<GradFn> residual_grad_fn;
    std::size_t element_count = 0;

    ResidualAddGradFn();

    void capture_inputs(Tensor& x, Tensor& r, cudaStream_t stream);

    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
