#pragma once
//======================================================//
//  ConcatGradFn.hpp
//  Backward node for row-wise feature concatenation.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ConcatGradFn : public GradFn {
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
    int rows = 0;
    int D1 = 0;
    int D2 = 0;

    ConcatGradFn();

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
