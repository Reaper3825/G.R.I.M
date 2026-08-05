#pragma once
//======================================================//
//  AddScalarGradFn.hpp
//  Backward node for adding a constant scalar: y = x + c.
//  Backward is a pure pass-through — c contributes no gradient.
//
//  Owns:
//    - struct AddScalarGradFn     (declared here)
//    - kernel_add_scalar_forward  (defined in AddScalarGradFn.cu)
//    - autograd::add_scalar(...)  (forward op; defined in AddScalarGradFn.cu)
//
//  Backward path uses TensorContract/GradientAccumulation.hpp; do not add
//  per-TU kernel_accumulate_grad copies here.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct AddScalarGradFn : public GradFn {
    bool input_requires_grad = false;
    std::shared_ptr<Tensor> input_gradient;
    std::size_t count = 0;

    AddScalarGradFn();
    ~AddScalarGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
