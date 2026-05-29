#pragma once
//======================================================//
//  ElementwiseMulGradFn.hpp
//  Backward node for elementwise multiply (Hadamard product).
//  Forward:  y = a ⊙ b
//  Backward: grad_a = grad_y ⊙ b
//            grad_b = grad_y ⊙ a
//
//  MEMORY OPTIMIZATION: backward holds NON-OWNING references to a/b
//  (cached_a / cached_b). Safe because ModelForwardOutputs owns the retained
//  per-layer tensors directly, and Issue #56 guarantees those inputs persist
//  until after backward completes.
//
//  Owns:
//    - struct ElementwiseMulGradFn   (declared here)
//    - kernel_elementwise_mul_forward / _backward (defined in .cu)
//    - autograd::elementwise_mul(...)            (defined in .cu)
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ElementwiseMulGradFn : public GradFn {
    bool a_requires_grad = false;
    bool b_requires_grad = false;
    float* a_grad = nullptr;
    float* b_grad = nullptr;
    std::shared_ptr<float> owned_a_grad;
    std::shared_ptr<float> owned_b_grad;
    TensorContract::TensorShape a_shape;
    TensorContract::TensorShape b_shape;
    std::shared_ptr<GradFn> a_grad_fn;
    std::shared_ptr<GradFn> b_grad_fn;

    const float* cached_a = nullptr;
    const float* cached_b = nullptr;
    std::size_t cached_size = 0;

    ElementwiseMulGradFn();

    void capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream);
    void set_cache_refs(const float* a_data, const float* b_data, std::size_t size);

    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
