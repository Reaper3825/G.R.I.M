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
//  Backward path uses the shared kernel_accumulate_grad pattern (also
//  defined in this TU under internal linkage to keep .cu self-contained).
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct AddScalarGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    std::size_t count = 0;

    AddScalarGradFn();
    ~AddScalarGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
