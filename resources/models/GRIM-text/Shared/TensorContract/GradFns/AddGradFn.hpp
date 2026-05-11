#pragma once
//======================================================//
//  AddGradFn.hpp
//  Backward node for elementwise add: c = a + b.
//  Backward: grad_a += grad_c, grad_b += grad_c (both pass-through).
//
//  Owns:
//    - struct AddGradFn        (declared here)
//    - autograd::add(...)      (forward op; defined in AddGradFn.cu)
//
//  Backward path uses TensorContract/GradientAccumulation.hpp; do not add
//  per-TU kernel_accumulate_grad copies here.
//
//  NOTE: AddGradFn is also reused by autograd::dropout's identity-edge
//  path (single-input pass-through), via capture_single_input(...).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct AddGradFn : public GradFn {
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    float* grad_a = nullptr;
    float* grad_b = nullptr;
    std::shared_ptr<float> owned_grad_a;
    std::shared_ptr<float> owned_grad_b;
    TensorContract::TensorShape a_shape;
    TensorContract::TensorShape b_shape;
    std::shared_ptr<GradFn> a_grad_fn;
    std::shared_ptr<GradFn> b_grad_fn;
    std::size_t element_count = 0;

    AddGradFn();

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream);
    void capture_single_input(Tensor& a, cudaStream_t stream);

    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
