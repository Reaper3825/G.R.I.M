#pragma once
//======================================================//
//  ExpGradFn.hpp
//  Backward node for element-wise exp: y = exp(x).
//
//  Owns:
//    - struct ExpGradFn           (declared here)
//    - kernel_exp_forward         (defined in ExpGradFn.cu)
//    - kernel_exp_backward        (defined in ExpGradFn.cu)
//    - autograd::exp(...)         (forward op; defined in ExpGradFn.cu)
//
//  Saves output (not input) — d/dx exp(x) = exp(x) = y, so the forward
//  result is the cheapest thing to keep around for backward.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ExpGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_output = nullptr;          ///< Non-owning ref into result.data
    std::size_t cached_size = 0;

    ExpGradFn();
    ~ExpGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void save_output(const float* output_data, std::size_t size);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
